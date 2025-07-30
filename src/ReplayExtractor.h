/* This file is part of the Spring engine (GPL v2 or later), see LICENSE.html */

#ifndef _CHANRTS_REPLAY_EXTRACTOR_H
#define _CHANRTS_REPLAY_EXTRACTOR_H

#include <string>
#include <vector>
#include <memory>
#include <fstream>

namespace chanrts {

// Forward declarations
struct GameState;
struct MLAction;
struct TrainingData;
class CChanRTS;

/**
 * Extracts training data from Spring replay files
 */
class ReplayExtractor {
public:
    ReplayExtractor(CChanRTS* ai);
    ~ReplayExtractor();
    
    // Main extraction methods
    bool IsReplayMode() const;
    void SetReplayMode(bool enabled, const std::string& replayPath = "");
    void ProcessReplayFrame(int frame, const GameState& state);
    void FinalizeReplayExtraction();
    
    // Batch processing
    static void ProcessReplayDirectory(const std::string& replayDir, const std::string& outputDir);
    static void ProcessSingleReplay(const std::string& replayPath, const std::string& outputPath);
    
    // Data export
    void SaveReplayData(const std::string& outputPath);
    void ExportToCSV(const std::string& csvPath);
    void ExportToBinary(const std::string& binPath); // For faster loading
    
    // Analysis helpers
    void AnalyzePlayerBehavior(int playerId);
    void ExtractUnitCommandSequences();
    void IdentifyStrategicPhases();
    
private:
    CChanRTS* parentAI;
    bool replayMode;
    std::string currentReplayPath;
    
    // Extracted data
    std::vector<TrainingData> replayData;
    
    // Player tracking
    struct PlayerInfo {
        int playerId;
        int teamId;
        std::string playerName;
        bool isAI;
        bool won;
        std::vector<int> unitIds;
    };
    std::vector<PlayerInfo> players;
    
    // Replay metadata
    struct ReplayMetadata {
        std::string mapName;
        std::string modName;
        int gameLength;
        int winnerTeam;
        std::string gameEndReason;
        float avgFrameTime;
    };
    ReplayMetadata metadata;
    
    // Helper methods
    void DetectPlayers();
    void AnalyzeGameOutcome();
    void CalculateReplayRewards();
    void ExtractCommandSequences(const GameState& state);
    
    // File I/O
    void WriteMetadata(std::ofstream& file);
    void WriteGameState(std::ofstream& file, const GameState& state);
};

/**
 * Utility class for batch replay processing
 */
class BatchReplayProcessor {
public:
    BatchReplayProcessor(const std::string& replayDir, const std::string& outputDir);
    
    void ProcessAllReplays();
    void ProcessReplaySubset(int startIndex, int count);
    void GenerateSummaryReport();
    
    // Filtering options
    void SetMapFilter(const std::vector<std::string>& allowedMaps);
    void SetPlayerFilter(const std::vector<std::string>& playerNames);
    void SetMinGameLength(int minFrames);
    void SetMaxGameLength(int maxFrames);
    
private:
    std::string replayDirectory;
    std::string outputDirectory;
    
    // Filters
    std::vector<std::string> mapFilter;
    std::vector<std::string> playerFilter;
    int minGameLength;
    int maxGameLength;
    
    // Processing stats
    int totalReplays;
    int processedReplays;
    int failedReplays;
    
    std::vector<std::string> FindReplayFiles();
    bool ShouldProcessReplay(const std::string& replayPath);
};

} // namespace chanrts

#endif // _CHANRTS_REPLAY_EXTRACTOR_H