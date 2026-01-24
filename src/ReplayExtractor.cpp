/* This file is part of the Spring engine (GPL v2 or later), see LICENSE.html */

#include "ReplayExtractor.h"
#include "ChanRTS.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <chrono>

namespace chanrts {

// Constructor
ReplayExtractor::ReplayExtractor(CChanRTS* ai) : 
    parentAI(ai), 
    replayMode(false),
    currentReplayPath("")
{
    // Initialize metadata
    metadata.gameLength = 0;
    metadata.winnerTeam = -1;
    metadata.avgFrameTime = 0.0f;
}

// Destructor
ReplayExtractor::~ReplayExtractor() {
    if (replayMode && !replayData.empty()) {
        FinalizeReplayExtraction();
    }
}

// Check if we're in replay mode
bool ReplayExtractor::IsReplayMode() const {
    return replayMode;
}

// Enable/disable replay mode
void ReplayExtractor::SetReplayMode(bool enabled, const std::string& replayPath) {
    replayMode = enabled;
    currentReplayPath = replayPath;
    
    if (enabled) {
        replayData.clear();
        players.clear();
        
        // Initialize replay metadata
        if (!replayPath.empty()) {
            size_t lastSlash = replayPath.find_last_of("/\\");
            if (lastSlash != std::string::npos) {
                std::string filename = replayPath.substr(lastSlash + 1);
                // Extract info from filename if possible
                // Format often: YYYYMMDD_HHMMSS_MapName_Players.sdfz
                metadata.mapName = filename; // Simplified
            }
        }
    }
}

// Process each frame of the replay
void ReplayExtractor::ProcessReplayFrame(int frame, const GameState& state) {
    if (!replayMode) return;
    
    // Sample every 900 frames (30 seconds) to avoid too much data
    if (frame % 900 != 0) return;
    
    TrainingData sample;
    sample.state = state;
    sample.sampleFrame = frame;
    sample.reward = 0.0f; // Will be calculated later
    sample.gameWon = false; // Unknown until game ends
    
    // Extract command sequences and player behavior
    ExtractCommandSequences(state);
    
    replayData.push_back(sample);
    metadata.gameLength = frame;
}

// Finalize replay extraction when game ends
void ReplayExtractor::FinalizeReplayExtraction() {
    if (replayData.empty()) return;
    
    AnalyzeGameOutcome();
    CalculateReplayRewards();
    
    // Auto-save to default location
    std::string outputPath = "/data/chanrts/trainingdata/replay_" + 
                           std::to_string(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())) + 
                           ".csv";
    SaveReplayData(outputPath);
}

// Analyze game outcome from final state
void ReplayExtractor::AnalyzeGameOutcome() {
    if (replayData.empty()) return;
    
    const GameState& finalState = replayData.back().state;
    
    // Simple heuristic: team with more units probably won
    // TODO: Improve with actual game outcome detection
    int friendlyUnits = finalState.friendlyUnitIds.size();
    int enemyUnits = finalState.enemyUnitIds.size();
    
    bool gameWon = (friendlyUnits > enemyUnits);
    metadata.winnerTeam = gameWon ? 0 : 1; // Simplified
    metadata.gameEndReason = gameWon ? "Victory" : "Defeat";
}

// Calculate rewards for all samples based on game outcome
void ReplayExtractor::CalculateReplayRewards() {
    if (replayData.empty()) return;
    
    bool gameWon = (metadata.winnerTeam == 0); // Assuming team 0 is us
    
    for (auto& sample : replayData) {
        sample.gameWon = gameWon;
        
        // Time-discounted reward
        float timeFactor = 1.0f - (sample.sampleFrame / float(metadata.gameLength));
        float baseReward = gameWon ? 1.0f : -1.0f;
        
        // Add behavioral rewards
        float behaviorReward = 0.0f;
        
        // Reward economic growth
        if (sample.state.metalIncome > 0 || sample.state.energyIncome > 0) {
            behaviorReward += 0.1f;
        }
        
        // Reward unit diversity
        if (sample.state.militaryUnitCount > 0 && sample.state.economicUnitCount > 0) {
            behaviorReward += 0.1f;
        }
        
        // Penalty for idle units
        int idleUnits = 0;
        for (const auto& unit : sample.state.allUnits) {
            if (!unit.isEnemy && unit.isIdle) {
                idleUnits++;
            }
        }
        if (idleUnits > sample.state.totalUnitCount * 0.3f) {
            behaviorReward -= 0.1f;
        }
        
        sample.reward = (baseReward * timeFactor) + behaviorReward;
    }
}

// Extract command sequences from game state
void ReplayExtractor::ExtractCommandSequences(const GameState& state) {
    // Track unit command patterns for behavioral analysis
    // This would be expanded to capture actual command sequences
    
    // For now, just count active vs idle units
    int activeUnits = 0;
    for (const auto& unit : state.allUnits) {
        if (!unit.isEnemy && !unit.isIdle) {
            activeUnits++;
        }
    }
    
    // Could store command patterns for sequence learning
}

// Save replay data to file
void ReplayExtractor::SaveReplayData(const std::string& outputPath) {
    ExportToCSV(outputPath);
}

// Export to CSV format
void ReplayExtractor::ExportToCSV(const std::string& csvPath) {
    if (replayData.empty()) return;
    
    try {
        // Ensure directory exists
        std::filesystem::path path(csvPath);
        std::filesystem::create_directories(path.parent_path());
        
        std::ofstream file(csvPath);
        if (!file.is_open()) {
            std::cerr << "Failed to open replay output file: " << csvPath << std::endl;
            return;
        }
        
        WriteMetadata(file);
        
        // Write CSV header
        file << "frame,game_progress,map_width,map_height,total_units,military_units,"
             << "economic_units,metal,energy,metal_income,energy_income,"
             << "active_units,idle_units,reward,game_won,replay_source\n";
        
        // Write data samples
        for (const auto& sample : replayData) {
            int idleUnits = 0;
            int activeUnits = 0;
            
            for (const auto& unit : sample.state.allUnits) {
                if (!unit.isEnemy) {
                    if (unit.isIdle) idleUnits++;
                    else activeUnits++;
                }
            }
            
            file << sample.sampleFrame << ","
                 << sample.state.gameProgress << ","
                 << sample.state.mapWidth << ","
                 << sample.state.mapHeight << ","
                 << sample.state.totalUnitCount << ","
                 << sample.state.militaryUnitCount << ","
                 << sample.state.economicUnitCount << ","
                 << sample.state.metal << ","
                 << sample.state.energy << ","
                 << sample.state.metalIncome << ","
                 << sample.state.energyIncome << ","
                 << activeUnits << ","
                 << idleUnits << ","
                 << sample.reward << ","
                 << (sample.gameWon ? 1 : 0) << ","
                 << currentReplayPath << "\n";
        }
        
        file.close();
        
        std::cout << "Replay data exported: " << csvPath 
                  << " (" << replayData.size() << " samples)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error exporting replay data: " << e.what() << std::endl;
    }
}

// Write metadata header to file
void ReplayExtractor::WriteMetadata(std::ofstream& file) {
    file << "# Replay Metadata\n";
    file << "# Map: " << metadata.mapName << "\n";
    file << "# Game Length: " << metadata.gameLength << " frames\n";
    file << "# Winner: Team " << metadata.winnerTeam << "\n";
    file << "# End Reason: " << metadata.gameEndReason << "\n";
    file << "# Source: " << currentReplayPath << "\n";
    file << "# Extracted: " << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\n";
    file << "#\n";
}

// Static method to process single replay
void ReplayExtractor::ProcessSingleReplay(const std::string& replayPath, const std::string& outputPath) {
    // This would require launching Spring with the replay file
    // For now, this is a placeholder for the batch processing system
    
    std::cout << "Processing replay: " << replayPath << " -> " << outputPath << std::endl;
    
    // TODO: Implement actual replay processing
    // This would involve:
    // 1. Launching Spring in headless mode with the replay
    // 2. Having ChanRTS extract data during replay playback
    // 3. Saving the extracted data to the output path
}

// Static method to process replay directory
void ReplayExtractor::ProcessReplayDirectory(const std::string& replayDir, const std::string& outputDir) {
    BatchReplayProcessor processor(replayDir, outputDir);
    processor.ProcessAllReplays();
}

// BatchReplayProcessor implementation
BatchReplayProcessor::BatchReplayProcessor(const std::string& replayDir, const std::string& outputDir) :
    replayDirectory(replayDir),
    outputDirectory(outputDir),
    minGameLength(600),  // 20 seconds minimum
    maxGameLength(18000), // 10 minutes maximum
    totalReplays(0),
    processedReplays(0),
    failedReplays(0)
{
    // Ensure output directory exists
    std::filesystem::create_directories(outputDirectory);
}

void BatchReplayProcessor::ProcessAllReplays() {
    auto replayFiles = FindReplayFiles();
    totalReplays = replayFiles.size();
    
    std::cout << "Found " << totalReplays << " replay files in " << replayDirectory << std::endl;
    
    for (const auto& replayPath : replayFiles) {
        if (ShouldProcessReplay(replayPath)) {
            try {
                std::string filename = std::filesystem::path(replayPath).stem().string();
                std::string outputPath = outputDirectory + "/" + filename + "_training.csv";
                
                ReplayExtractor::ProcessSingleReplay(replayPath, outputPath);
                processedReplays++;
                
            } catch (const std::exception& e) {
                std::cerr << "Failed to process replay " << replayPath << ": " << e.what() << std::endl;
                failedReplays++;
            }
        }
    }
    
    GenerateSummaryReport();
}

std::vector<std::string> BatchReplayProcessor::FindReplayFiles() {
    std::vector<std::string> replayFiles;
    
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(replayDirectory)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                
                if (extension == ".sdfz" || extension == ".demo") {
                    replayFiles.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error scanning replay directory: " << e.what() << std::endl;
    }
    
    return replayFiles;
}

bool BatchReplayProcessor::ShouldProcessReplay(const std::string& replayPath) {
    // Apply filters here
    // For now, just basic filename filtering
    std::string filename = std::filesystem::path(replayPath).filename().string();
    
    // Skip if map filter is set and doesn't match
    if (!mapFilter.empty()) {
        bool mapMatches = false;
        for (const auto& allowedMap : mapFilter) {
            if (filename.find(allowedMap) != std::string::npos) {
                mapMatches = true;
                break;
            }
        }
        if (!mapMatches) return false;
    }
    
    return true;
}

void BatchReplayProcessor::GenerateSummaryReport() {
    std::string reportPath = outputDirectory + "/processing_summary.txt";
    std::ofstream report(reportPath);
    
    if (report.is_open()) {
        report << "Batch Replay Processing Summary\n";
        report << "==============================\n\n";
        report << "Source Directory: " << replayDirectory << "\n";
        report << "Output Directory: " << outputDirectory << "\n";
        report << "Total Replays Found: " << totalReplays << "\n";
        report << "Successfully Processed: " << processedReplays << "\n";
        report << "Failed: " << failedReplays << "\n";
        report << "Success Rate: " << (totalReplays > 0 ? (processedReplays * 100.0 / totalReplays) : 0) << "%\n";
        
        report.close();
        std::cout << "Summary report saved: " << reportPath << std::endl;
    }
}

} // namespace chanrts