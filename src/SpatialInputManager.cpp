#include "SpatialInputManager.h"
#include "ChanRTS.h"

#include <algorithm>
#include <cmath>
#include <memory>

namespace chanrts {

SpatialInputManager::SpatialInputManager(springai::OOAICallback* callback)
    : callback(callback)
    , map(nullptr)
    , game(nullptr)
    , economy(nullptr)
    , metalResource(nullptr)
    , energyResource(nullptr)
    , lastUpdateFrame(0)
    , staticLayersInitialized(false)
    , ourTeamId(-1)
{
    // Initialize Spring API interfaces
    if (callback) {
        map = callback->GetMap();
        game = callback->GetGame();
        economy = callback->GetEconomy();
        
        // Get resource references
        auto resources = callback->GetResources();
        for (springai::Resource* resource : resources) {
            std::string name = resource->GetName();
            if (name == "Metal") {
                metalResource = resource;
            } else if (name == "Energy") {
                energyResource = resource;
            } else {
                delete resource;
            }
        }
        
        ourTeamId = 0;  // Default team ID - Spring API changed
        
        // Initialize tensor dimensions based on map size
        if (map) {
            int mapWidth = map->GetWidth();
            int mapHeight = map->GetHeight();
            currentInputTensor = SpatialTensor(mapWidth, mapHeight);
            nextInputTensor = SpatialTensor(mapWidth, mapHeight);
            
            // Initialize tracking arrays
            recentCombatTracker.resize(mapHeight, std::vector<float>(mapWidth, 0.0f));
            movementTracker.resize(mapHeight, std::vector<float>(mapWidth, 0.0f));
            constructionTracker.resize(mapHeight, std::vector<float>(mapWidth, 0.0f));
        }
        
        BuildUnitTypeMapping();
        UpdateTeamAlliances();
    }
}

SpatialInputManager::~SpatialInputManager() {
    delete map;
    delete game;
    delete economy;
    delete metalResource;
    delete energyResource;
}

void SpatialInputManager::UpdateInputTensor(const GameState& state) {
    std::lock_guard<std::mutex> lock(inputMutex);
    
    // 1. Clear all layers
    ClearAllLayers(nextInputTensor);
    
    // 2. Initialize static layers if needed
    if (!staticLayersInitialized) {
        InitializeStaticLayers();
        staticLayersInitialized = true;
    }
    
    // 3. Populate unit type and ownership layers
    PopulateUnitLayers(state, nextInputTensor);
    
    // 4. Populate dynamic unit state
    PopulateUnitStateLayers(state, nextInputTensor);
    
    // 5. Populate static map features (cached)
    PopulateStaticMapFeatures(nextInputTensor);
    
    // 6. Update visibility layers
    PopulateVisibilityLayers(nextInputTensor);
    
    // 7. Update resource information (uniform across map)
    PopulateResourceLayers(nextInputTensor);
    
    // 8. Calculate strategic overlays
    PopulateStrategicOverlays(state, nextInputTensor);
    
    // 9. Update command and control layers
    PopulateCommandControlLayers(state, nextInputTensor);
    
    // 10. Swap buffers
    currentInputTensor = nextInputTensor;
    lastUpdateFrame = state.currentFrame;
}

SpatialTensor SpatialInputManager::GetInputTensorForInference() {
    std::lock_guard<std::mutex> lock(inputMutex);
    return currentInputTensor;  // Returns a copy for thread safety
}

void SpatialInputManager::InitializeStaticLayers() {
    if (!map) return;
    
    int width = map->GetWidth();
    int height = map->GetHeight();
    
    // Cache height map
    heightMapCache.clear();
    heightMapCache.reserve(width * height);
    for (int z = 0; z < height; ++z) {
        for (int x = 0; x < width; ++x) {
            float heightValue = map->GetElevationAt(x, z);
            heightMapCache.push_back(heightValue);
        }
    }
    
    // Cache slope map (calculated from height differences)
    slopeMapCache.clear();
    slopeMapCache.reserve(width * height);
    for (int z = 0; z < height; ++z) {
        for (int x = 0; x < width; ++x) {
            float slope = 0.0f;
            if (x > 0 && x < width - 1 && z > 0 && z < height - 1) {
                float h1 = map->GetElevationAt(x - 1, z);
                float h2 = map->GetElevationAt(x + 1, z);
                float h3 = map->GetElevationAt(x, z - 1);
                float h4 = map->GetElevationAt(x, z + 1);
                slope = std::sqrt((h2 - h1) * (h2 - h1) + (h4 - h3) * (h4 - h3)) / 2.0f;
            }
            slopeMapCache.push_back(slope);
        }
    }
    
    // Cache metal map
    metalMapCache.clear();
    metalMapCache.reserve(width * height);
    for (int z = 0; z < height; ++z) {
        for (int x = 0; x < width; ++x) {
            float metalValue = 0.5f;  // Default metal value - GetMetalAmount API changed
            metalMapCache.push_back(metalValue);
        }
    }
    
    // Cache water areas
    waterMapCache.clear();
    waterMapCache.reserve(width * height);
    for (int z = 0; z < height; ++z) {
        for (int x = 0; x < width; ++x) {
            bool isWater = (map->GetElevationAt(x, z) < 0.0f);
            waterMapCache.push_back(isWater);
        }
    }
}

void SpatialInputManager::BuildUnitTypeMapping() {
    // This would be populated with actual BAR unit definitions
    // For now, we'll create a placeholder mapping system
    
    // TODO: Load actual unit definitions from BAR and map them to layer indices
    // This should iterate through all UnitDefs and assign layer indices based on:
    // - Faction (Armada: 0-263, Cortex: 264-532, Legion: 533-747, etc.)
    // - Alphabetical ordering within faction
    
    // Placeholder examples:
    unitNameToLayerIndex["armcom"] = 0;           // ARM Commander
    unitNameToLayerIndex["armatlas"] = 1;         // ARM Atlas Transport
    unitNameToLayerIndex["armca"] = 2;            // ARM Construction Aircraft
    // ... continue for all 888 unit types
    
    unitNameToLayerIndex["corcom"] = 264;         // COR Commander
    unitNameToLayerIndex["corvalk"] = 265;        // COR Valkyrie Transport
    // ... continue for all Cortex units
    
    unitNameToLayerIndex["legcom"] = 533;         // LEG Commander
    // ... continue for all Legion units
    
    // Build reverse mapping from defId to layer index
    if (callback) {
        auto unitDefs = callback->GetUnitDefs();
        for (springai::UnitDef* unitDef : unitDefs) {
            std::string unitName = unitDef->GetName();
            auto it = unitNameToLayerIndex.find(unitName);
            if (it != unitNameToLayerIndex.end()) {
                unitDefToLayerIndex[unitDef->GetUnitDefId()] = it->second;
            }
            delete unitDef;
        }
    }
}

void SpatialInputManager::UpdateTeamAlliances() {
    if (!callback) return;
    
    alliedTeamIds.clear();
    enemyTeamIds.clear();
    
    // Get all teams and determine alliances
    int numTeams = callback->GetNumTeams();
    for (int i = 0; i < numTeams; ++i) {
        if (i == ourTeamId) {
            // Skip our own team
            continue;
        }
        // Simplified team classification - assume all others are enemies
        enemyTeamIds.push_back(i);
    }
}

void SpatialInputManager::ClearAllLayers(SpatialTensor& tensor) {
    for (auto& layer : tensor.layers) {
        std::fill(layer.begin(), layer.end(), 0.0f);
    }
}

void SpatialInputManager::PopulateUnitLayers(const GameState& state, SpatialTensor& tensor) {
    // Clear unit-related layers
    for (int layer = SpatialLayers::UNIT_TYPE_START; layer <= SpatialLayers::WRECKAGE_INDICATOR; ++layer) {
        std::fill(tensor.layers[layer].begin(), tensor.layers[layer].end(), 0.0f);
    }
    
    for (const auto& unitInfo : state.allUnits) {
        int x = static_cast<int>(unitInfo.x);
        int z = static_cast<int>(unitInfo.z);
        if (x < 0 || x >= tensor.width || z < 0 || z >= tensor.height) continue;
        
        int cellIndex = z * tensor.width + x;
        
        // Set unit type position
        int unitTypeLayer = GetUnitTypeLayer(unitInfo.defId);
        if (unitTypeLayer >= 0 && unitTypeLayer <= SpatialLayers::UNIT_TYPE_END) {
            tensor.layers[unitTypeLayer][cellIndex] = 1.0f;
        }
        
        // Set ownership
        int ownershipLayer = GetOwnershipLayer(unitInfo.team);
        tensor.layers[ownershipLayer][cellIndex] = 1.0f;
    }
}

void SpatialInputManager::PopulateUnitStateLayers(const GameState& state, SpatialTensor& tensor) {
    for (const auto& unitInfo : state.allUnits) {
        int x = static_cast<int>(unitInfo.x);
        int z = static_cast<int>(unitInfo.z);
        if (x < 0 || x >= tensor.width || z < 0 || z >= tensor.height) continue;
        
        int cellIndex = z * tensor.width + x;
        
        // Health ratio
        if (unitInfo.maxHealth > 0) {
            tensor.layers[SpatialLayers::UNIT_HEALTH_RATIO][cellIndex] = 
                unitInfo.health / unitInfo.maxHealth;
        }
        
        // Heading (normalize to -1 to 1)
        tensor.layers[SpatialLayers::UNIT_HEADING][cellIndex] = unitInfo.heading / M_PI;
        
        // Velocity components (normalized)
        tensor.layers[SpatialLayers::UNIT_VELOCITY_X][cellIndex] = 
            std::tanh(unitInfo.velX / 10.0f);  // Normalize to -1,1 range
        tensor.layers[SpatialLayers::UNIT_VELOCITY_Z][cellIndex] = 
            std::tanh(unitInfo.velZ / 10.0f);
        
        // Build progress
        tensor.layers[SpatialLayers::UNIT_BUILD_PROGRESS][cellIndex] = unitInfo.buildProgress;
        
        // Experience level (normalized)
        tensor.layers[SpatialLayers::UNIT_EXPERIENCE_LEVEL][cellIndex] = 
            std::min(unitInfo.experience / 1000.0f, 1.0f);
        
        // Command queue size (normalized)
        tensor.layers[SpatialLayers::UNIT_COMMAND_QUEUE_SIZE][cellIndex] = 
            std::min(unitInfo.commandCount / 10.0f, 1.0f);
        
        // Status flags (1.0 = true, 0.0 = false)
        tensor.layers[SpatialLayers::UNIT_IS_IDLE][cellIndex] = unitInfo.isIdle ? 1.0f : 0.0f;
        tensor.layers[SpatialLayers::UNIT_IS_MOVING][cellIndex] = unitInfo.isMoving ? 1.0f : 0.0f;
        tensor.layers[SpatialLayers::UNIT_IS_ATTACKING][cellIndex] = unitInfo.isAttacking ? 1.0f : 0.0f;
        tensor.layers[SpatialLayers::UNIT_IS_BUILDING][cellIndex] = unitInfo.isBuilding ? 1.0f : 0.0f;
        tensor.layers[SpatialLayers::UNIT_IS_BEING_BUILT][cellIndex] = unitInfo.isBeingBuilt ? 1.0f : 0.0f;
        tensor.layers[SpatialLayers::UNIT_IS_CLOAKED][cellIndex] = unitInfo.isCloaked ? 1.0f : 0.0f;
        tensor.layers[SpatialLayers::UNIT_IS_STUNNED][cellIndex] = unitInfo.isStunned ? 1.0f : 0.0f;
        tensor.layers[SpatialLayers::UNIT_IS_PARALYZED][cellIndex] = unitInfo.isParalyzed ? 1.0f : 0.0f;
    }
}

void SpatialInputManager::PopulateStaticMapFeatures(SpatialTensor& tensor) {
    if (heightMapCache.empty() || slopeMapCache.empty()) return;
    
    int width = tensor.width;
    int height = tensor.height;
    
    // Copy cached static data
    for (int i = 0; i < width * height && i < heightMapCache.size(); ++i) {
        // Normalize height (assuming reasonable height range)
        tensor.layers[SpatialLayers::TERRAIN_HEIGHT][i] = 
            std::tanh(heightMapCache[i] / 100.0f) * 0.5f + 0.5f;  // 0-1 range
        
        // Normalize slope
        tensor.layers[SpatialLayers::TERRAIN_SLOPE][i] = 
            std::min(slopeMapCache[i] / 45.0f, 1.0f);  // 0-1 range (45 degrees max)
        
        // Water depth/areas
        tensor.layers[SpatialLayers::WATER_AREAS][i] = waterMapCache[i] ? 1.0f : 0.0f;
        tensor.layers[SpatialLayers::WATER_DEPTH][i] = 
            waterMapCache[i] ? std::max(-heightMapCache[i] / 50.0f, 0.0f) : 0.0f;
        
        // Metal density
        tensor.layers[SpatialLayers::METAL_DENSITY][i] = 
            std::min(metalMapCache[i] / 255.0f, 1.0f);  // Assuming 0-255 range
        
        // Metal spots (high metal areas)
        tensor.layers[SpatialLayers::METAL_SPOTS][i] = 
            (metalMapCache[i] > 100.0f) ? 1.0f : 0.0f;  // Binary metal spot indicator
    }
}

void SpatialInputManager::PopulateVisibilityLayers(SpatialTensor& tensor) {
    if (!map) return;
    
    int width = tensor.width;
    int height = tensor.height;
    
    // Get LOS and radar maps from Spring API
    auto losMap = map->GetLosMap();
    auto radarMap = map->GetRadarMap();
    
    for (int z = 0; z < height; ++z) {
        for (int x = 0; x < width; ++x) {
            int index = z * width + x;
            if (index < losMap.size() && index < radarMap.size()) {
                // LOS coverage (1.0 = visible, 0.0 = not visible)
                tensor.layers[SpatialLayers::FRIENDLY_LOS][index] = 
                    (losMap[index] > 0) ? 1.0f : 0.0f;
                
                // Radar coverage
                tensor.layers[SpatialLayers::FRIENDLY_RADAR][index] = 
                    (radarMap[index] > 0) ? 1.0f : 0.0f;
                
                // Fog of war (inverse of LOS)
                tensor.layers[SpatialLayers::FOG_OF_WAR][index] = 
                    (losMap[index] > 0) ? 0.0f : 1.0f;
            }
        }
    }
}

void SpatialInputManager::PopulateResourceLayers(SpatialTensor& tensor) {
    float metalLevel, energyLevel, metalIncome, energyIncome;
    float metalUsage, energyUsage, metalStorage, energyStorage;
    
    GetResourceData(metalLevel, energyLevel, metalIncome, energyIncome,
                   metalUsage, energyUsage, metalStorage, energyStorage);
    
    // Fill entire map uniformly with resource values
    std::fill(tensor.layers[SpatialLayers::PLAYER_METAL_LEVEL].begin(),
              tensor.layers[SpatialLayers::PLAYER_METAL_LEVEL].end(), metalLevel);
    std::fill(tensor.layers[SpatialLayers::PLAYER_ENERGY_LEVEL].begin(),
              tensor.layers[SpatialLayers::PLAYER_ENERGY_LEVEL].end(), energyLevel);
    std::fill(tensor.layers[SpatialLayers::PLAYER_METAL_INCOME].begin(),
              tensor.layers[SpatialLayers::PLAYER_METAL_INCOME].end(), metalIncome);
    std::fill(tensor.layers[SpatialLayers::PLAYER_ENERGY_INCOME].begin(),
              tensor.layers[SpatialLayers::PLAYER_ENERGY_INCOME].end(), energyIncome);
    std::fill(tensor.layers[SpatialLayers::PLAYER_METAL_USAGE].begin(),
              tensor.layers[SpatialLayers::PLAYER_METAL_USAGE].end(), metalUsage);
    std::fill(tensor.layers[SpatialLayers::PLAYER_ENERGY_USAGE].begin(),
              tensor.layers[SpatialLayers::PLAYER_ENERGY_USAGE].end(), energyUsage);
    std::fill(tensor.layers[SpatialLayers::PLAYER_METAL_STORAGE].begin(),
              tensor.layers[SpatialLayers::PLAYER_METAL_STORAGE].end(), metalStorage);
    std::fill(tensor.layers[SpatialLayers::PLAYER_ENERGY_STORAGE].begin(),
              tensor.layers[SpatialLayers::PLAYER_ENERGY_STORAGE].end(), energyStorage);
    
    // Get ally resource totals
    float allyMetalTotal = 0.0f, allyEnergyTotal = 0.0f;
    float allyMetalIncome = 0.0f, allyEnergyIncome = 0.0f;
    
    // Simplified ally resource calculation - GetTeam API changed
    // allyMetalTotal = alliedTeamIds.size() * 1000.0f;  // Placeholder
    // allyEnergyTotal = alliedTeamIds.size() * 1000.0f;
    
    // Normalize and fill ally resource layers
    std::fill(tensor.layers[SpatialLayers::ALLY_TOTAL_METAL_LEVEL].begin(),
              tensor.layers[SpatialLayers::ALLY_TOTAL_METAL_LEVEL].end(),
              std::min(allyMetalTotal / 50000.0f, 1.0f));
    std::fill(tensor.layers[SpatialLayers::ALLY_TOTAL_ENERGY_LEVEL].begin(),
              tensor.layers[SpatialLayers::ALLY_TOTAL_ENERGY_LEVEL].end(),
              std::min(allyEnergyTotal / 100000.0f, 1.0f));
}

void SpatialInputManager::PopulateStrategicOverlays(const GameState& state, SpatialTensor& tensor) {
    CalculateControlZones(state, tensor);
    CalculateThreatLevels(state, tensor);
    UpdateRecentCombat(state, tensor);
    CalculateStrategicValue(tensor);
    CalculateFrontlines(tensor);
    UpdateConstructionActivity(state, tensor);
}

void SpatialInputManager::PopulateCommandControlLayers(const GameState& state, SpatialTensor& tensor) {
    int width = tensor.width;
    int height = tensor.height;
    
    // Reset command control layers
    std::fill(tensor.layers[SpatialLayers::COMMAND_QUEUE_DENSITY].begin(),
              tensor.layers[SpatialLayers::COMMAND_QUEUE_DENSITY].end(), 0.0f);
    std::fill(tensor.layers[SpatialLayers::IDLE_UNIT_DENSITY].begin(),
              tensor.layers[SpatialLayers::IDLE_UNIT_DENSITY].end(), 0.0f);
    
    // Populate based on unit states
    for (const auto& unitInfo : state.allUnits) {
        if (unitInfo.isEnemy) continue;  // Only our units
        
        int x = static_cast<int>(unitInfo.x);
        int z = static_cast<int>(unitInfo.z);
        if (x < 0 || x >= width || z < 0 || z >= height) continue;
        
        int cellIndex = z * width + x;
        
        // Command queue density
        if (unitInfo.commandCount > 0) {
            tensor.layers[SpatialLayers::COMMAND_QUEUE_DENSITY][cellIndex] += 
                std::min(unitInfo.commandCount / 10.0f, 1.0f);
        }
        
        // Idle unit density
        if (unitInfo.isIdle) {
            tensor.layers[SpatialLayers::IDLE_UNIT_DENSITY][cellIndex] += 0.1f;
        }
        
        // Construction progress
        if (unitInfo.isBeingBuilt && unitInfo.buildProgress > 0) {
            tensor.layers[SpatialLayers::CONSTRUCTION_PROGRESS][cellIndex] = unitInfo.buildProgress;
        }
        
        // Factory production (if unit is a factory with build queue)
        if (unitInfo.isFactory && !unitInfo.buildQueue.empty()) {
            tensor.layers[SpatialLayers::FACTORY_PRODUCTION][cellIndex] = 1.0f;
        }
    }
    
    // Normalize density layers
    NormalizeLayer(tensor.layers[SpatialLayers::COMMAND_QUEUE_DENSITY], 5.0f);
    NormalizeLayer(tensor.layers[SpatialLayers::IDLE_UNIT_DENSITY], 2.0f);
}

// Strategic overlay calculation methods (simplified implementations)

void SpatialInputManager::CalculateControlZones(const GameState& state, SpatialTensor& tensor) {
    // TODO: Implement distance-based influence calculation
    // This would use unit positions and weapon ranges to calculate control areas
}

void SpatialInputManager::CalculateThreatLevels(const GameState& state, SpatialTensor& tensor) {
    // TODO: Implement threat assessment based on enemy firepower
    // This would convolve enemy unit positions with their weapon ranges and damage
}

void SpatialInputManager::UpdateRecentCombat(const GameState& state, SpatialTensor& tensor) {
    // TODO: Implement combat tracking with decay over time
    // Track unit deaths and damage events, apply exponential decay
}

void SpatialInputManager::CalculateStrategicValue(SpatialTensor& tensor) {
    // TODO: Calculate strategic importance based on resources, chokepoints, etc.
    // Combine metal spots, terrain features, and map position importance
}

void SpatialInputManager::CalculateFrontlines(SpatialTensor& tensor) {
    // TODO: Calculate battle front based on friendly vs enemy control zone gradients
}

void SpatialInputManager::UpdateConstructionActivity(const GameState& state, SpatialTensor& tensor) {
    // Already handled in PopulateCommandControlLayers
}

// Utility methods

int SpatialInputManager::GetUnitTypeLayer(int unitDefId) const {
    auto it = unitDefToLayerIndex.find(unitDefId);
    return (it != unitDefToLayerIndex.end()) ? it->second : -1;
}

int SpatialInputManager::GetOwnershipLayer(int teamId, bool isWreckage) const {
    if (isWreckage) return SpatialLayers::WRECKAGE_INDICATOR;
    if (teamId == ourTeamId) return SpatialLayers::PLAYER_OWNERSHIP;
    if (IsAllyTeam(teamId)) return SpatialLayers::ALLY_OWNERSHIP;
    if (IsEnemyTeam(teamId)) return SpatialLayers::ENEMY_OWNERSHIP;
    return SpatialLayers::NEUTRAL_OWNERSHIP;
}

bool SpatialInputManager::IsAllyTeam(int teamId) const {
    return std::find(alliedTeamIds.begin(), alliedTeamIds.end(), teamId) != alliedTeamIds.end();
}

bool SpatialInputManager::IsEnemyTeam(int teamId) const {
    return std::find(enemyTeamIds.begin(), enemyTeamIds.end(), teamId) != enemyTeamIds.end();
}

void SpatialInputManager::NormalizeLayer(std::vector<float>& layer, float maxValue) const {
    for (float& value : layer) {
        value = std::min(value / maxValue, 1.0f);
    }
}

float SpatialInputManager::CalculateDistance(float x1, float z1, float x2, float z2) const {
    float dx = x2 - x1;
    float dz = z2 - z1;
    return std::sqrt(dx * dx + dz * dz);
}

void SpatialInputManager::GetResourceData(float& metalLevel, float& energyLevel,
                                         float& metalIncome, float& energyIncome,
                                         float& metalUsage, float& energyUsage,
                                         float& metalStorage, float& energyStorage) const {
    if (!economy || !metalResource || !energyResource) {
        // Default values if API unavailable
        metalLevel = energyLevel = 0.5f;
        metalIncome = energyIncome = 0.1f;
        metalUsage = energyUsage = 0.05f;
        metalStorage = energyStorage = 1.0f;
        return;
    }
    
    // Get real Spring API resource data
    metalLevel = std::min(economy->GetCurrent(metalResource) / 10000.0f, 1.0f);
    energyLevel = std::min(economy->GetCurrent(energyResource) / 10000.0f, 1.0f);
    metalIncome = std::min(economy->GetIncome(metalResource) / 100.0f, 1.0f);
    energyIncome = std::min(economy->GetIncome(energyResource) / 200.0f, 1.0f);
    metalUsage = std::min(economy->GetUsage(metalResource) / 100.0f, 1.0f);
    energyUsage = std::min(economy->GetUsage(energyResource) / 200.0f, 1.0f);
    metalStorage = std::min(economy->GetStorage(metalResource) / 10000.0f, 1.0f);
    energyStorage = std::min(economy->GetStorage(energyResource) / 10000.0f, 1.0f);
}

} // namespace chanrts