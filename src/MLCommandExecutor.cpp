#include "MLCommandExecutor.h"
#include "ChanRTS.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace chanrts {

MLCommandExecutor::MLCommandExecutor(springai::OOAICallback* callback)
    : callback(callback)
    , spatialCommandThreshold(0.3f)
    , buildSelectionThreshold(0.4f)
    , unitSelectionThreshold(0.2f)
    , maxCommandsPerFrame(10)
    , mapWidth(512)
    , mapHeight(512)
    , pixelToWorldX(1.0f)
    , pixelToWorldY(1.0f)
    , commandsExecutedThisFrame(0)
    , totalCommandsExecuted(0)
{
    buildMapper = std::make_unique<HierarchicalBuildMapper>();
    
    // Initialize map dimensions from Spring API
    if (callback) {
        auto map = callback->GetMap();
        if (map) {
            mapWidth = map->GetWidth();
            mapHeight = map->GetHeight();
            pixelToWorldX = static_cast<float>(mapWidth) / 512.0f;   // Assuming 512x512 neural net input
            pixelToWorldY = static_cast<float>(mapHeight) / 512.0f;
            delete map;
        }
        
        // Detect current faction for build mapping
        // Use a simple default approach since GetTeams() API changed
        buildMapper->SetCurrentFaction("arm");  // Default to ARM
    }
}

MLCommandExecutor::~MLCommandExecutor() = default;

void MLCommandExecutor::ExecuteMLOutput(const MLOutput& output, const GameState& gameState) {
    commandsExecutedThisFrame = 0;
    
    // 1. Select units to command based on unit selection scores
    auto allUnits = callback->GetFriendlyUnits();
    auto selectedUnits = SelectUnitsForCommands(output.unitSelectionScores, allUnits);
    
    if (selectedUnits.empty()) {
        // Clean up unit pointers
        for (auto unit : allUnits) delete unit;
        return;
    }
    
    // 2. Extract spatial commands from neural network output
    auto spatialCommands = ExtractSpatialCommands(output.spatialCommands);
    
    // 3. Extract hierarchical build commands
    auto buildCommands = ExtractBuildCommands(output.buildCategorySelection, output.buildUnitInCategory);
    
    // 4. Apply global strategic parameters
    ApplyGlobalParameters(output.globalParams);
    
    // 5. Execute spatial commands (movement, attack, patrol)
    bool queueCommands = (output.queueCommands > 0.5f);
    ExecuteSpatialCommands(spatialCommands, selectedUnits, queueCommands);
    
    // 6. Execute build commands with available builders
    auto builders = FilterBuilders(allUnits);
    ExecuteBuildCommands(buildCommands, builders, queueCommands);
    
    totalCommandsExecuted += commandsExecutedThisFrame;
    
    // Clean up unit pointers
    for (auto unit : allUnits) delete unit;
}

void MLCommandExecutor::SetExecutionThresholds(float spatialThresh, float buildThresh, float unitThresh) {
    spatialCommandThreshold = spatialThresh;
    buildSelectionThreshold = buildThresh;
    unitSelectionThreshold = unitThresh;
}

void MLCommandExecutor::UpdateMapDimensions(int width, int height) {
    mapWidth = width;
    mapHeight = height;
    pixelToWorldX = static_cast<float>(width) / 512.0f;
    pixelToWorldY = static_cast<float>(height) / 512.0f;
}

std::vector<springai::Unit*> MLCommandExecutor::SelectUnitsForCommands(
    const std::vector<float>& selectionScores,
    const std::vector<springai::Unit*>& allUnits) const {
    
    std::vector<springai::Unit*> selectedUnits;
    
    // Create pairs of (unit, score) for sorting
    std::vector<std::pair<springai::Unit*, float>> unitScores;
    
    for (size_t i = 0; i < allUnits.size() && i < selectionScores.size(); ++i) {
        if (selectionScores[i] > unitSelectionThreshold) {
            unitScores.emplace_back(allUnits[i], selectionScores[i]);
        }
    }
    
    // Sort by selection score (highest first)
    std::sort(unitScores.begin(), unitScores.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Take top units up to reasonable limit
    int maxUnits = std::min(20, static_cast<int>(unitScores.size()));
    for (int i = 0; i < maxUnits; ++i) {
        selectedUnits.push_back(unitScores[i].first);
    }
    
    return selectedUnits;
}

std::vector<SpatialCommand> MLCommandExecutor::ExtractSpatialCommands(
    const std::vector<std::vector<float>>& spatialMaps) const {
    
    std::vector<SpatialCommand> commands;
    
    // Process each spatial command type
    for (int cmdType = 0; cmdType < SPATIAL_COMMAND_COUNT; ++cmdType) {
        if (cmdType >= spatialMaps.size()) continue;
        
        const auto& probabilityMap = spatialMaps[cmdType];
        auto highProbLocations = FindHighProbabilityLocations(probabilityMap, spatialCommandThreshold);
        
        // Convert high-probability pixels to world coordinates
        for (const auto& location : highProbLocations) {
            int pixelX = location.first;
            int pixelY = location.second;
            int index = pixelY * 512 + pixelX;  // Assuming 512x512 input
            
            if (index < probabilityMap.size()) {
                float probability = probabilityMap[index];
                auto worldCoords = PixelToWorldCoordinates(pixelX, pixelY);
                
                SpatialCommand cmd(static_cast<SpatialCommandType>(cmdType),
                                 worldCoords.first, 0.0f, worldCoords.second,
                                 probability, pixelX, pixelY);
                
                if (ValidateCommand(cmd)) {
                    commands.push_back(cmd);
                }
            }
        }
    }
    
    // Prioritize and limit commands
    PrioritizeCommands(commands);
    if (commands.size() > maxCommandsPerFrame / 2) {  // Reserve half for build commands
        commands.resize(maxCommandsPerFrame / 2);
    }
    
    return commands;
}

std::vector<HierarchicalBuildCommand> MLCommandExecutor::ExtractBuildCommands(
    const std::vector<std::vector<float>>& buildCategoryMaps,
    const std::vector<std::vector<float>>& buildUnitSelection) const {
    
    std::vector<HierarchicalBuildCommand> commands;
    
    // Process each build category
    for (int category = 0; category < BUILD_CATEGORY_COUNT; ++category) {
        if (category >= buildCategoryMaps.size()) continue;
        
        const auto& categoryMap = buildCategoryMaps[category];
        auto highProbLocations = FindHighProbabilityLocations(categoryMap, buildSelectionThreshold);
        
        // For each high-probability location, determine specific unit to build
        for (const auto& location : highProbLocations) {
            int pixelX = location.first;
            int pixelY = location.second;
            int index = pixelY * 512 + pixelX;
            
            if (index < categoryMap.size()) {
                float categoryProb = categoryMap[index];
                auto worldCoords = PixelToWorldCoordinates(pixelX, pixelY);
                
                // Find best unit within this category
                int bestUnitIndex = 0;
                float bestUnitProb = 0.0f;
                
                if (category < buildUnitSelection.size()) {
                    const auto& unitProbs = buildUnitSelection[category];
                    for (size_t i = 0; i < unitProbs.size(); ++i) {
                        if (unitProbs[i] > bestUnitProb) {
                            bestUnitProb = unitProbs[i];
                            bestUnitIndex = static_cast<int>(i);
                        }
                    }
                }
                
                // Create hierarchical build command
                HierarchicalBuildCommand cmd;
                cmd.x = worldCoords.first;
                cmd.y = 0.0f;
                cmd.z = worldCoords.second;
                cmd.category = static_cast<BuildCategory>(category);
                cmd.unitIndexInCategory = bestUnitIndex;
                cmd.categoryProbability = categoryProb;
                cmd.unitProbability = bestUnitProb;
                cmd.combinedProbability = categoryProb * bestUnitProb;
                cmd.sourcePixelX = pixelX;
                cmd.sourcePixelY = pixelY;
                cmd.unitDefName = ResolveBuildUnitDef(cmd.category, cmd.unitIndexInCategory);
                
                if (ValidateBuildCommand(cmd)) {
                    commands.push_back(cmd);
                }
            }
        }
    }
    
    // Prioritize and limit build commands
    PrioritizeBuildCommands(commands);
    if (commands.size() > maxCommandsPerFrame / 2) {
        commands.resize(maxCommandsPerFrame / 2);
    }
    
    return commands;
}

void MLCommandExecutor::ExecuteSpatialCommands(const std::vector<SpatialCommand>& commands,
                                              const std::vector<springai::Unit*>& selectedUnits,
                                              bool queueCommands) {
    
    for (const auto& cmd : commands) {
        if (commandsExecutedThisFrame >= maxCommandsPerFrame) break;
        
        // Filter units appropriate for this command type
        auto suitableUnits = FilterUnitsForCommand(selectedUnits, cmd.type);
        if (suitableUnits.empty()) continue;
        
        // Execute the appropriate Spring API command
        switch (cmd.type) {
            case MOVE_TO:
                ExecuteMoveCommand(suitableUnits, cmd.x, cmd.z, queueCommands);
                break;
                
            case PATROL_TO:
                ExecutePatrolCommand(suitableUnits, cmd.x, cmd.z, queueCommands);
                break;
                
            case ATTACK_GROUND:
                ExecuteAttackGroundCommand(suitableUnits, cmd.x, cmd.z, queueCommands);
                break;
                
            case FIGHT_TO:
                // Fight command is similar to move but with aggressive stance
                ExecuteMoveCommand(suitableUnits, cmd.x, cmd.z, queueCommands);
                break;
                
            case SCOUT_AREA: {
                // Send fast units to scout
                auto scouts = FilterUnitsForCommand(suitableUnits, SCOUT_AREA);
                ExecuteMoveCommand(scouts, cmd.x, cmd.z, queueCommands);
                break;
            }
                
            default:
                // Other command types handled elsewhere or not implemented
                break;
        }
        
        commandsExecutedThisFrame++;
    }
}

void MLCommandExecutor::ExecuteBuildCommands(const std::vector<HierarchicalBuildCommand>& commands,
                                            const std::vector<springai::Unit*>& builders,
                                            bool queueCommands) {
    
    // Filter to idle builders for new construction
    auto availableBuilders = FilterIdleUnits(builders);
    if (availableBuilders.empty()) return;
    
    size_t builderIndex = 0;
    for (const auto& cmd : commands) {
        if (commandsExecutedThisFrame >= maxCommandsPerFrame) break;
        if (builderIndex >= availableBuilders.size()) break;
        
        springai::Unit* builder = availableBuilders[builderIndex];
        
        // Check if this builder can construct the requested unit
        if (CanUnitBuild(builder, cmd.unitDefName)) {
            ExecuteBuildCommand(builder, cmd.unitDefName, cmd.x, cmd.z, queueCommands);
            commandsExecutedThisFrame++;
            builderIndex = (builderIndex + 1) % availableBuilders.size();
        }
    }
}

void MLCommandExecutor::ApplyGlobalParameters(const std::vector<float>& globalParams) {
    // Global parameters influence overall AI behavior
    // These would typically be stored and used by other systems
    
    if (globalParams.size() >= GLOBAL_PARAM_COUNT) {
        float combatPriority = globalParams[FACTORY_PRIORITY_COMBAT];
        float economyPriority = globalParams[FACTORY_PRIORITY_ECONOMY];
        float scoutPriority = globalParams[FACTORY_PRIORITY_SCOUTS];
        
        float defensivePosture = globalParams[DEFENSIVE_POSTURE];
        float aggressivePosture = globalParams[AGGRESSIVE_POSTURE];
        float expansionPriority = globalParams[EXPANSION_PRIORITY];
        
        // These parameters would influence:
        // - Factory production queues
        // - Resource allocation priorities
        // - Unit behavior settings
        // - Strategic decision making
        
        // For now, just log the parameters (in actual implementation, 
        // these would be stored and used by other AI subsystems)
    }
}

// Private helper methods

std::vector<std::pair<int, int>> MLCommandExecutor::FindHighProbabilityLocations(
    const std::vector<float>& probabilityMap, 
    float threshold) const {
    
    std::vector<std::pair<int, int>> locations;
    
    // Simple peak detection - find local maxima above threshold
    for (int y = 1; y < 511; ++y) {  // Assuming 512x512 input
        for (int x = 1; x < 511; ++x) {
            int index = y * 512 + x;
            if (index >= probabilityMap.size()) continue;
            
            float value = probabilityMap[index];
            if (value < threshold) continue;
            
            // Check if this is a local maximum
            bool isLocalMax = true;
            for (int dy = -1; dy <= 1 && isLocalMax; ++dy) {
                for (int dx = -1; dx <= 1 && isLocalMax; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    
                    int neighborIndex = (y + dy) * 512 + (x + dx);
                    if (neighborIndex < probabilityMap.size() && 
                        probabilityMap[neighborIndex] > value) {
                        isLocalMax = false;
                    }
                }
            }
            
            if (isLocalMax) {
                locations.emplace_back(x, y);
            }
        }
    }
    
    return locations;
}

std::pair<float, float> MLCommandExecutor::PixelToWorldCoordinates(int pixelX, int pixelY) const {
    float worldX = pixelX * pixelToWorldX;
    float worldZ = pixelY * pixelToWorldY;
    return {worldX, worldZ};
}

bool MLCommandExecutor::IsLocationValid(float x, float z, SpatialCommandType commandType) const {
    // Basic bounds checking
    if (x < 0 || z < 0 || x >= mapWidth || z >= mapHeight) {
        return false;
    }
    
    // Command-specific validation
    switch (commandType) {
        case BUILD_STRUCTURE:
        case BUILD_FACTORY:
        case BUILD_DEFENSE:
        case BUILD_ECONOMY:
            // Building locations need to be on land and not blocked
            return !NearWater(x, z);  // Simplified check
            
        default:
            return true;  // Most commands can be issued anywhere
    }
}

bool MLCommandExecutor::IsLocationValidForBuilding(float x, float z, BuildCategory category) const {
    switch (category) {
        case METAL_EXTRACTION:
            return NearMetalSpot(x, z);
            
        case BASIC_SHIPYARD:
        case ADVANCED_SHIPYARD:
            return NearWater(x, z);
            
        case BASIC_DEFENSES:
        case HEAVY_DEFENSES:
        case AA_DEFENSES:
            return InFrontlineArea(x, z) || HasNearbyEnemies(x, z);
            
        default:
            return !NearWater(x, z);  // Most buildings need land
    }
}

std::vector<springai::Unit*> MLCommandExecutor::FilterUnitsForCommand(
    const std::vector<springai::Unit*>& units,
    SpatialCommandType commandType) const {
    
    std::vector<springai::Unit*> filtered;
    
    for (springai::Unit* unit : units) {
        if (!unit) continue;
        
        switch (commandType) {
            case MOVE_TO:
            case PATROL_TO:
            case FIGHT_TO:
                // Any mobile unit can move
                if (unit->GetMaxSpeed() > 0.1f) {
                    filtered.push_back(unit);
                }
                break;
                
            case ATTACK_GROUND:
                // Only units with weapons
                if (unit->GetMaxRange() > 0) {
                    filtered.push_back(unit);
                }
                break;
                
            case SCOUT_AREA:
                // Fast, cheap units for scouting
                if (unit->GetMaxSpeed() > 2.0f) {
                    filtered.push_back(unit);
                }
                break;
                
            default:
                filtered.push_back(unit);
                break;
        }
    }
    
    return filtered;
}

std::vector<springai::Unit*> MLCommandExecutor::FilterBuilders(
    const std::vector<springai::Unit*>& units) const {
    
    std::vector<springai::Unit*> builders;
    for (springai::Unit* unit : units) {
        if (unit && unit->GetDef()) {
            auto unitDef = unit->GetDef();
            if (unitDef->IsBuilder()) {
                builders.push_back(unit);
            }
            delete unitDef;
        }
    }
    return builders;
}

std::vector<springai::Unit*> MLCommandExecutor::FilterIdleUnits(
    const std::vector<springai::Unit*>& units) const {
    
    std::vector<springai::Unit*> idleUnits;
    for (springai::Unit* unit : units) {
        if (unit && unit->GetVel().x < 0.1f && unit->GetVel().z < 0.1f) {
            // Simplified idle detection based on velocity
            idleUnits.push_back(unit);
        }
    }
    return idleUnits;
}

// Spring API command execution methods

void MLCommandExecutor::ExecuteMoveCommand(const std::vector<springai::Unit*>& units,
                                          float x, float z, bool queueCommand) {
    for (springai::Unit* unit : units) {
        if (unit) {
            springai::AIFloat3 position = {x, 0.0f, z};
            // Note: In actual implementation, would need proper command options
            unit->MoveTo(position);
        }
    }
}

void MLCommandExecutor::ExecutePatrolCommand(const std::vector<springai::Unit*>& units,
                                            float x, float z, bool queueCommand) {
    for (springai::Unit* unit : units) {
        if (unit) {
            springai::AIFloat3 position = {x, 0.0f, z};
            unit->PatrolTo(position);
        }
    }
}

void MLCommandExecutor::ExecuteAttackGroundCommand(const std::vector<springai::Unit*>& units,
                                                  float x, float z, bool queueCommand) {
    for (springai::Unit* unit : units) {
        if (unit && unit->GetMaxRange() > 0) {
            springai::AIFloat3 position = {x, 0.0f, z};
            // Attack ground - simplified implementation
            unit->Fight(position);  // Fight is closest equivalent
        }
    }
}

void MLCommandExecutor::ExecuteBuildCommand(springai::Unit* builder,
                                           const std::string& unitDefName,
                                           float x, float z, bool queueCommand) {
    if (!builder) return;
    
    // Get unit definition by name
    auto unitDefs = callback->GetUnitDefs();
    springai::UnitDef* targetDef = nullptr;
    
    for (springai::UnitDef* def : unitDefs) {
        if (def && def->GetName() == unitDefName) {
            targetDef = def;
            break;
        }
    }
    
    if (targetDef) {
        springai::AIFloat3 position = {x, 0.0f, z};
        builder->Build(targetDef, position, 0);  // facing = 0
    }
    
    // Clean up unit defs
    for (springai::UnitDef* def : unitDefs) {
        if (def != targetDef) delete def;
    }
    delete targetDef;
}

// Build type resolution and validation

std::string MLCommandExecutor::ResolveBuildUnitDef(BuildCategory category, int unitIndex) const {
    return buildMapper->GetUnitDefName(category, unitIndex);
}

bool MLCommandExecutor::CanUnitBuild(springai::Unit* unit, const std::string& unitDefName) const {
    if (!unit || unitDefName.empty()) return false;
    
    // Check if unit is a builder
    auto unitDef = unit->GetDef();
    if (!unitDef || !unitDef->IsBuilder()) {
        delete unitDef;
        return false;
    }
    delete unitDef;
    
    // In full implementation, would check build options
    // For now, assume all builders can build basic structures
    return true;
}

// Context analysis methods (simplified implementations)

bool MLCommandExecutor::NearMetalSpot(float x, float z) const {
    // Would check actual metal map data
    return true;  // Placeholder
}

bool MLCommandExecutor::NearWater(float x, float z) const {
    // Would check actual height map for water areas
    return false;  // Placeholder
}

bool MLCommandExecutor::InFrontlineArea(float x, float z) const {
    // Would analyze unit positions to determine frontlines
    return false;  // Placeholder
}

bool MLCommandExecutor::HasNearbyEnemies(float x, float z, float radius) const {
    // Would check for enemy units within radius
    return false;  // Placeholder
}

// Command validation and prioritization

bool MLCommandExecutor::ValidateCommand(const SpatialCommand& cmd) const {
    return IsLocationValid(cmd.x, cmd.z, cmd.type) && cmd.probability > spatialCommandThreshold;
}

bool MLCommandExecutor::ValidateBuildCommand(const HierarchicalBuildCommand& cmd) const {
    return !cmd.unitDefName.empty() && 
           cmd.combinedProbability > buildSelectionThreshold &&
           IsLocationValidForBuilding(cmd.x, cmd.z, cmd.category);
}

void MLCommandExecutor::PrioritizeCommands(std::vector<SpatialCommand>& commands) const {
    // Sort by probability (highest first)
    std::sort(commands.begin(), commands.end(),
             [](const SpatialCommand& a, const SpatialCommand& b) {
                 return a.probability > b.probability;
             });
}

void MLCommandExecutor::PrioritizeBuildCommands(std::vector<HierarchicalBuildCommand>& commands) const {
    // Sort by combined probability (highest first)
    std::sort(commands.begin(), commands.end(),
             [](const HierarchicalBuildCommand& a, const HierarchicalBuildCommand& b) {
                 return a.combinedProbability > b.combinedProbability;
             });
}

} // namespace chanrts