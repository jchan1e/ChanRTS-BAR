#ifndef _CHANRTS_SPATIALINPUTMANAGER_H
#define _CHANRTS_SPATIALINPUTMANAGER_H

#include "OOAICallback.h"
#include "Unit.h"
#include "UnitDef.h"
#include "Map.h"
#include "Game.h"
#include "Economy.h"
#include "Resource.h"
#include "Team.h"

#include <vector>
#include <mutex>
#include <unordered_map>
#include <memory>

namespace chanrts {

// Forward declarations
struct GameState;

/**
 * Spatial tensor for ML model input - 968 channels at map resolution
 */
struct SpatialTensor {
    static const int CHANNELS = 968;
    
    int width;
    int height;
    int channels;
    
    // Primary data structure: layers[channel][spatial_index]
    std::vector<std::vector<float>> layers;
    
    // Alternative flat data structure for compatibility
    std::vector<float> data;
    
    SpatialTensor() : width(0), height(0), channels(CHANNELS) {}
    
    SpatialTensor(int w, int h) : width(w), height(h), channels(CHANNELS) {
        layers.resize(channels);
        int spatialSize = width * height;
        for (int i = 0; i < channels; ++i) {
            layers[i].resize(spatialSize, 0.0f);
        }
        data.resize(channels * spatialSize, 0.0f);
    }
    
    // Copy constructor
    SpatialTensor(const SpatialTensor& other) 
        : width(other.width), height(other.height), channels(other.channels),
          layers(other.layers), data(other.data) {}
    
    // Assignment operator
    SpatialTensor& operator=(const SpatialTensor& other) {
        if (this != &other) {
            width = other.width;
            height = other.height;
            channels = other.channels;
            layers = other.layers;
            data = other.data;
        }
        return *this;
    }
    
    // Utility methods
    void Clear() {
        for (auto& layer : layers) {
            std::fill(layer.begin(), layer.end(), 0.0f);
        }
        std::fill(data.begin(), data.end(), 0.0f);
    }
    
    bool IsValid() const {
        return width > 0 && height > 0 && 
               layers.size() == channels &&
               (!layers.empty() && layers[0].size() == width * height);
    }
};

/**
 * Layer indices for the 968-channel spatial input tensor
 */
namespace SpatialLayers {
    // Unit Type Position Layers (0-887)
    const int UNIT_TYPE_START = 0;
    const int UNIT_TYPE_END = 887;
    
    // Ownership Indicator Layers (888-892)
    const int PLAYER_OWNERSHIP = 888;
    const int ALLY_OWNERSHIP = 889;
    const int ENEMY_OWNERSHIP = 890;
    const int NEUTRAL_OWNERSHIP = 891;
    const int WRECKAGE_INDICATOR = 892;
    
    // Dynamic Unit State Layers (893-909)
    const int UNIT_HEALTH_RATIO = 893;
    const int UNIT_HEADING = 894;
    const int UNIT_VELOCITY_X = 895;
    const int UNIT_VELOCITY_Z = 896;
    const int UNIT_BUILD_PROGRESS = 897;
    const int UNIT_EXPERIENCE_LEVEL = 898;
    const int UNIT_COMMAND_QUEUE_SIZE = 899;
    const int UNIT_IS_IDLE = 900;
    const int UNIT_IS_MOVING = 901;
    const int UNIT_IS_ATTACKING = 902;
    const int UNIT_IS_BUILDING = 903;
    const int UNIT_IS_BEING_BUILT = 904;
    const int UNIT_IS_CLOAKED = 905;
    const int UNIT_IS_STUNNED = 906;
    const int UNIT_IS_PARALYZED = 907;
    const int UNIT_IS_JAMMED = 908;
    const int UNIT_SPECIAL_ABILITY_READY = 909;
    
    // Static Map Features (910-925)
    const int TERRAIN_HEIGHT = 910;
    const int TERRAIN_SLOPE = 911;
    const int WATER_DEPTH = 912;
    const int METAL_SPOTS = 913;
    const int METAL_DENSITY = 914;
    const int GEOTHERMAL_VENTS = 915;
    const int IMPASSABLE_TERRAIN = 916;
    const int WATER_AREAS = 917;
    const int CLIFF_AREAS = 918;
    const int STRATEGIC_POINTS = 919;
    const int SPAWN_POINTS = 920;
    const int NO_BUILD_ZONES = 921;
    const int LAVA_AREAS = 922;
    const int RESTRICTED_AIRSPACE = 923;
    const int TELEPORTER_ZONES = 924;
    const int MAP_BOUNDARIES = 925;
    
    // Visibility and Intelligence (926-933)
    const int FRIENDLY_LOS = 926;
    const int FRIENDLY_RADAR = 927;
    const int ENEMY_LOS_ESTIMATE = 928;
    const int ENEMY_RADAR_ESTIMATE = 929;
    const int FOG_OF_WAR = 930;
    const int LAST_SEEN_TIMER = 931;
    const int JAMMING_COVERAGE = 932;
    const int STEALTH_DETECTION = 933;
    
    // Resource Information (934-949)
    const int PLAYER_METAL_LEVEL = 934;
    const int PLAYER_ENERGY_LEVEL = 935;
    const int PLAYER_METAL_INCOME = 936;
    const int PLAYER_ENERGY_INCOME = 937;
    const int PLAYER_METAL_USAGE = 938;
    const int PLAYER_ENERGY_USAGE = 939;
    const int ALLY_TOTAL_METAL_LEVEL = 940;
    const int ALLY_TOTAL_ENERGY_LEVEL = 941;
    const int ALLY_TOTAL_METAL_INCOME = 942;
    const int ALLY_TOTAL_ENERGY_INCOME = 943;
    const int PLAYER_METAL_STORAGE = 944;
    const int PLAYER_ENERGY_STORAGE = 945;
    const int PLAYER_METAL_PULL = 946;
    const int PLAYER_ENERGY_PULL = 947;
    const int RESOURCE_EFFICIENCY = 948;
    const int OVERDRIVE_EFFICIENCY = 949;
    
    // Strategic Overlays (950-961)
    const int FRIENDLY_CONTROL_ZONES = 950;
    const int ENEMY_CONTROL_ZONES = 951;
    const int CONTESTED_ZONES = 952;
    const int STRATEGIC_VALUE = 953;
    const int THREAT_LEVEL = 954;
    const int RECENT_COMBAT = 955;
    const int CONSTRUCTION_ACTIVITY = 956;
    const int RESOURCE_FLOW = 957;
    const int PATHFINDING_FLOW = 958;
    const int FRONTLINE_INDICATOR = 959;
    const int UNIT_DENSITY_GRADIENT = 960;
    const int AVERAGE_UNIT_EXPERIENCE = 961;
    
    // Command and Control Information (962-967)
    const int COMMAND_QUEUE_DENSITY = 962;
    const int CONSTRUCTION_PROGRESS = 963;
    const int FACTORY_PRODUCTION = 964;
    const int RECENT_ORDERS = 965;
    const int IDLE_UNIT_DENSITY = 966;
    const int STALLED_CONSTRUCTION = 967;
    
    const int TOTAL_CHANNELS = 968;
}

/**
 * Manages spatial input tensor generation for ML inference
 */
class SpatialInputManager {
private:
    // Spring API interfaces
    springai::OOAICallback* callback;
    springai::Map* map;
    springai::Game* game;
    springai::Economy* economy;
    springai::Resource* metalResource;
    springai::Resource* energyResource;
    
    // Tensor management
    SpatialTensor currentInputTensor;
    SpatialTensor nextInputTensor;
    std::mutex inputMutex;
    
    // Unit type mapping
    std::unordered_map<int, int> unitDefToLayerIndex;
    std::unordered_map<std::string, int> unitNameToLayerIndex;
    
    // Strategic overlay calculation state
    std::vector<std::vector<float>> recentCombatTracker;
    std::vector<std::vector<float>> movementTracker;
    std::vector<std::vector<float>> constructionTracker;
    int lastUpdateFrame;
    bool staticLayersInitialized;
    
    // Team and alliance tracking
    int ourTeamId;
    std::vector<int> alliedTeamIds;
    std::vector<int> enemyTeamIds;
    
    // Cached static data
    std::vector<float> heightMapCache;
    std::vector<float> slopeMapCache;
    std::vector<float> metalMapCache;
    std::vector<bool> waterMapCache;
    
public:
    SpatialInputManager(springai::OOAICallback* callback);
    ~SpatialInputManager();
    
    // Main interface
    void UpdateInputTensor(const GameState& state);
    SpatialTensor GetInputTensorForInference();
    bool HasNewData() const { return lastUpdateFrame > 0; }
    
    // Initialization
    void InitializeStaticLayers();
    void BuildUnitTypeMapping();
    void UpdateTeamAlliances();
    
private:
    // Layer population methods
    void ClearAllLayers(SpatialTensor& tensor);
    void PopulateUnitLayers(const GameState& state, SpatialTensor& tensor);
    void PopulateUnitStateLayers(const GameState& state, SpatialTensor& tensor);
    void PopulateStaticMapFeatures(SpatialTensor& tensor);
    void PopulateVisibilityLayers(SpatialTensor& tensor);
    void PopulateResourceLayers(SpatialTensor& tensor);
    void PopulateStrategicOverlays(const GameState& state, SpatialTensor& tensor);
    void PopulateCommandControlLayers(const GameState& state, SpatialTensor& tensor);
    
    // Strategic overlay calculations
    void CalculateControlZones(const GameState& state, SpatialTensor& tensor);
    void CalculateThreatLevels(const GameState& state, SpatialTensor& tensor);
    void UpdateRecentCombat(const GameState& state, SpatialTensor& tensor);
    void CalculateStrategicValue(SpatialTensor& tensor);
    void CalculateFrontlines(SpatialTensor& tensor);
    void UpdateConstructionActivity(const GameState& state, SpatialTensor& tensor);
    
    // Utility methods
    int GetUnitTypeLayer(int unitDefId) const;
    int GetOwnershipLayer(int teamId, bool isWreckage = false) const;
    bool IsAllyTeam(int teamId) const;
    bool IsEnemyTeam(int teamId) const;
    void NormalizeLayer(std::vector<float>& layer, float maxValue) const;
    float CalculateDistance(float x1, float z1, float x2, float z2) const;
    void ApplyGaussianBlur(std::vector<float>& layer, int width, int height, float sigma) const;
    
    // Spring API helper methods
    std::vector<springai::Unit*> GetAllUnits() const;
    std::vector<springai::Team*> GetAlliedTeams() const;
    void GetResourceData(float& metalLevel, float& energyLevel, 
                        float& metalIncome, float& energyIncome,
                        float& metalUsage, float& energyUsage,
                        float& metalStorage, float& energyStorage) const;
};

} // namespace chanrts

#endif // _CHANRTS_SPATIALINPUTMANAGER_H