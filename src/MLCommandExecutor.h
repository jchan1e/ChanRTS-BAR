#ifndef _CHANRTS_MLCOMMANDEXECUTOR_H
#define _CHANRTS_MLCOMMANDEXECUTOR_H

#include "OOAICallback.h"
#include "Unit.h"
#include "UnitDef.h"
#include "MLConstants.h"
#include "HierarchicalBuildSystem.h"

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>

namespace chanrts {

// Forward declarations
struct GameState;

/**
 * Spatial command types for neural network output
 */
enum SpatialCommandType {
    // Movement Commands (target location on map)
    MOVE_TO = 0,                    // Move selected units to location
    PATROL_TO = 1,                  // Patrol to location  
    ATTACK_GROUND = 2,              // Attack ground at location
    FIGHT_TO = 3,                   // Fight (aggressive move) to location
    
    // Construction Commands (build location on map)  
    BUILD_STRUCTURE = 4,            // Build structure at location
    BUILD_FACTORY = 5,              // Build factory at location
    BUILD_DEFENSE = 6,              // Build defense at location
    BUILD_ECONOMY = 7,              // Build economic structure
    
    // Special Location Commands
    RECLAIM_AREA = 8,               // Reclaim resources in area
    TERRAFORM_AREA = 9,             // Terraform terrain at location
    SCOUT_AREA = 10                // Send scouts to explore area
};

/**
 * Global strategic parameters
 */
enum GlobalParameter {
    // Unit Production Control
    FACTORY_PRIORITY_COMBAT = 0,    // Focus on combat unit production
    FACTORY_PRIORITY_ECONOMY = 1,   // Focus on economic units
    FACTORY_PRIORITY_SCOUTS = 2,    // Focus on scout/support units
    
    // Strategic State
    DEFENSIVE_POSTURE = 3,          // Adopt defensive stance
    AGGRESSIVE_POSTURE = 4,         // Adopt aggressive stance  
    EXPANSION_PRIORITY = 5         // Focus on territorial expansion
};

/**
 * Extracted spatial command with location and probability
 */
struct SpatialCommand {
    SpatialCommandType type;
    float x, y, z;                  // World coordinates
    float probability;              // Neural network confidence
    int sourcePixelX, sourcePixelY; // Original pixel location
    
    SpatialCommand() : type(MOVE_TO), x(0), y(0), z(0), probability(0.0f),
                      sourcePixelX(0), sourcePixelY(0) {}
    
    SpatialCommand(SpatialCommandType t, float px, float py, float pz, float prob, int sx, int sy)
        : type(t), x(px), y(py), z(pz), probability(prob), sourcePixelX(sx), sourcePixelY(sy) {}
};

// MLOutput and HierarchicalBuildCommand are defined in HierarchicalBuildSystem.h

/**
 * Executes ML output by converting neural network predictions to Spring API commands
 */
class MLCommandExecutor {
private:
    springai::OOAICallback* callback;
    
    // Build type mappings for each faction
    std::unique_ptr<HierarchicalBuildMapper> buildMapper;
    
    // Execution parameters
    float spatialCommandThreshold;      // Minimum probability for spatial commands
    float buildSelectionThreshold;      // Minimum probability for build commands
    float unitSelectionThreshold;       // Minimum probability for unit selection
    int maxCommandsPerFrame;            // Limit commands per execution
    
    // Map dimensions for coordinate conversion
    int mapWidth, mapHeight;
    float pixelToWorldX, pixelToWorldY;
    
    // Performance tracking
    int commandsExecutedThisFrame;
    int totalCommandsExecuted;
    
public:
    MLCommandExecutor(springai::OOAICallback* callback);
    ~MLCommandExecutor();
    
    // Main execution interface
    void ExecuteMLOutput(const MLOutput& output, const GameState& gameState);
    
    // Configuration
    void SetExecutionThresholds(float spatialThresh, float buildThresh, float unitThresh);
    void SetMaxCommandsPerFrame(int maxCommands) { maxCommandsPerFrame = maxCommands; }
    void UpdateMapDimensions(int width, int height);
    
    // Command extraction methods
    std::vector<springai::Unit*> SelectUnitsForCommands(
        const std::vector<float>& selectionScores,
        const std::vector<springai::Unit*>& allUnits) const;
    
    std::vector<SpatialCommand> ExtractSpatialCommands(
        const std::vector<std::vector<float>>& spatialMaps) const;
    
    std::vector<HierarchicalBuildCommand> ExtractBuildCommands(
        const std::vector<std::vector<float>>& buildCategoryMaps,
        const std::vector<std::vector<float>>& buildUnitSelection) const;
    
    // Command execution methods
    void ExecuteSpatialCommands(const std::vector<SpatialCommand>& commands,
                               const std::vector<springai::Unit*>& selectedUnits,
                               bool queueCommands);
    
    void ExecuteBuildCommands(const std::vector<HierarchicalBuildCommand>& commands,
                             const std::vector<springai::Unit*>& builders,
                             bool queueCommands);
    
    void ApplyGlobalParameters(const std::vector<float>& globalParams);
    
    // Statistics and debugging
    int GetCommandsExecutedThisFrame() const { return commandsExecutedThisFrame; }
    int GetTotalCommandsExecuted() const { return totalCommandsExecuted; }
    void ResetFrameStats() { commandsExecutedThisFrame = 0; }
    
private:
    // Spatial analysis methods
    std::vector<std::pair<int, int>> FindHighProbabilityLocations(
        const std::vector<float>& probabilityMap, 
        float threshold) const;
    
    std::pair<float, float> PixelToWorldCoordinates(int pixelX, int pixelY) const;
    bool IsLocationValid(float x, float z, SpatialCommandType commandType) const;
    bool IsLocationValidForBuilding(float x, float z, BuildCategory category) const;
    
    // Unit filtering and selection
    std::vector<springai::Unit*> FilterUnitsForCommand(
        const std::vector<springai::Unit*>& units,
        SpatialCommandType commandType) const;
    
    std::vector<springai::Unit*> FilterBuilders(
        const std::vector<springai::Unit*>& units) const;
    
    std::vector<springai::Unit*> FilterIdleUnits(
        const std::vector<springai::Unit*>& units) const;
    
    // Spring API command execution
    void ExecuteMoveCommand(const std::vector<springai::Unit*>& units,
                           float x, float z, bool queueCommand);
    
    void ExecutePatrolCommand(const std::vector<springai::Unit*>& units,
                             float x, float z, bool queueCommand);
    
    void ExecuteAttackGroundCommand(const std::vector<springai::Unit*>& units,
                                   float x, float z, bool queueCommand);
    
    void ExecuteBuildCommand(springai::Unit* builder,
                            const std::string& unitDefName,
                            float x, float z, bool queueCommand);
    
    // Build type resolution
    std::string ResolveBuildUnitDef(BuildCategory category, int unitIndex) const;
    bool CanUnitBuild(springai::Unit* unit, const std::string& unitDefName) const;
    
    // Context analysis for better decision making
    bool NearMetalSpot(float x, float z) const;
    bool NearWater(float x, float z) const;
    bool InFrontlineArea(float x, float z) const;
    bool HasNearbyEnemies(float x, float z, float radius = 300.0f) const;
    
    // Command validation and filtering
    bool ValidateCommand(const SpatialCommand& cmd) const;
    bool ValidateBuildCommand(const HierarchicalBuildCommand& cmd) const;
    void PrioritizeCommands(std::vector<SpatialCommand>& commands) const;
    void PrioritizeBuildCommands(std::vector<HierarchicalBuildCommand>& commands) const;
};

} // namespace chanrts

#endif // _CHANRTS_MLCOMMANDEXECUTOR_H