#ifndef _CHANRTS_HIERARCHICALBUILDSYSTEM_H
#define _CHANRTS_HIERARCHICALBUILDSYSTEM_H

#include <vector>
#include <unordered_map>
#include <string>
#include "MLConstants.h"

namespace chanrts {

/**
 * Hierarchical Build Categories - Stage 1 Selection
 * These are broad functional categories that group similar buildings/units
 */
enum BuildCategory {
    // Economic Structures
    METAL_EXTRACTION = 0,           // Metal extractors, moho mines
    ENERGY_GENERATION = 1,          // Solar panels, wind generators, fusion
    ENERGY_STORAGE = 2,             // Energy storage structures
    METAL_STORAGE = 3,              // Metal storage structures
    METAL_MAKERS = 4,               // Metal makers, converters
    
    // Military Structures  
    BASIC_DEFENSES = 5,             // Light laser towers, machine guns
    HEAVY_DEFENSES = 6,             // Heavy laser towers, plasma cannons
    AA_DEFENSES = 7,                // Anti-air missiles, flak
    ARTILLERY_DEFENSES = 8,         // Long range cannons, bertha, annihilator
    SPECIAL_WEAPONS = 9,            // Nukes, anti-nukes, superweapons
    
    // Intelligence & Support
    RADARS = 10,                    // Radar towers, advanced radar
    JAMMERS = 11,                   // Radar jammers, stealth
    SONAR = 12,                     // Sonar stations, advanced sonar
    
    // Production Facilities
    BASIC_VEHICLE_FACTORY = 13,     // Vehicle plants, construction vehicles
    ADVANCED_VEHICLE_FACTORY = 14,  // Advanced vehicle plants, T2 vehicles
    BASIC_AIRCRAFT_FACTORY = 15,    // Aircraft plants, construction aircraft
    ADVANCED_AIRCRAFT_FACTORY = 16, // Advanced aircraft plants, T2 aircraft
    BASIC_SHIPYARD = 17,            // Shipyards, construction ships
    ADVANCED_SHIPYARD = 18,         // Advanced shipyards, T2 ships
    BASIC_BOT_FACTORY = 19,         // K-bot labs, construction bots
    ADVANCED_BOT_FACTORY = 20,      // Advanced K-bot labs, T2 bots
    SPECIAL_FACTORIES = 21,         // Seaplane platforms, hover platforms
    
    // Vehicles - Combat
    LIGHT_VEHICLES = 22,            // Scouts, light tanks, raiders
    MEDIUM_VEHICLES = 23,           // Main battle tanks, assault vehicles  
    HEAVY_VEHICLES = 24,            // Heavy tanks, siege units
    ARTILLERY_VEHICLES = 25,        // Mobile artillery, rocket launchers
    AA_VEHICLES = 26,               // Mobile AA, SAM launchers
    SUPPORT_VEHICLES = 27,          // Jammers, radars, repair vehicles
    
    // Aircraft - Combat
    FIGHTERS = 28,                  // Air superiority fighters, interceptors
    BOMBERS = 29,                   // Strategic bombers, dive bombers
    GUNSHIPS = 30,                  // Attack helicopters, gunships
    TRANSPORT_AIRCRAFT = 31,        // Transports, heavy lifters
    RECONNAISSANCE_AIRCRAFT = 32,   // Scouts, AWACS, spy planes
    
    // Naval - Combat  
    LIGHT_SHIPS = 33,               // Patrol boats, destroyers, subs
    HEAVY_SHIPS = 34,               // Battleships, heavy cruisers
    AIRCRAFT_CARRIERS = 35,         // Carriers, seaplane tenders
    SUPPORT_SHIPS = 36,             // Radar ships, jammers, repair
    
    // Bots - Combat
    LIGHT_BOTS = 37,                // Scout bots, raiders, infantry
    MEDIUM_BOTS = 38,               // Assault bots, main infantry
    HEAVY_BOTS = 39,                // Heavy assault bots, siege bots
    SPECIAL_BOTS = 40,              // Specialist units, unique bots
    
    // Construction & Utility
    BASIC_CONSTRUCTION = 41,        // Basic construction units (land/air/sea)
    ADVANCED_CONSTRUCTION = 42,     // Advanced construction units
    TRANSPORT_UTILITY = 43,         // Transports, cargo units
    REPAIR_UTILITY = 44,            // Repair units, nano-towers
    RECLAMATION_UTILITY = 45,       // Resurrection bots, reclaimers
    
    // Special Categories
    COMMANDERS = 46,                // Commanders, upgrades
    EXPERIMENTAL = 47,              // Experimental units, prototypes
    CAMPAIGN_SPECIAL = 48,          // Campaign-specific units
    DECORATIVE = 49,                // Trees, rocks, decorations
    
    BUILD_CATEGORY_COUNT = 50
};

/**
 * Maximum units per category - estimated based on BAR unit counts
 */
const int MAX_UNITS_PER_CATEGORY[BUILD_CATEGORY_COUNT] = {
    8,   // METAL_EXTRACTION (basic mex, moho, etc.)
    12,  // ENERGY_GENERATION (solar, wind, geo, fusion, etc.)
    4,   // ENERGY_STORAGE 
    4,   // METAL_STORAGE
    6,   // METAL_MAKERS
    
    15,  // BASIC_DEFENSES (LLT, HLT, various turrets)
    12,  // HEAVY_DEFENSES (plasma, laser, particle)
    10,  // AA_DEFENSES (missile towers, flak)
    8,   // ARTILLERY_DEFENSES (bertha, annihilator)
    8,   // SPECIAL_WEAPONS (nuke, anti-nuke, super)
    
    6,   // RADARS
    4,   // JAMMERS  
    4,   // SONAR
    
    3,   // BASIC_VEHICLE_FACTORY
    3,   // ADVANCED_VEHICLE_FACTORY
    3,   // BASIC_AIRCRAFT_FACTORY
    3,   // ADVANCED_AIRCRAFT_FACTORY
    3,   // BASIC_SHIPYARD
    3,   // ADVANCED_SHIPYARD
    3,   // BASIC_BOT_FACTORY
    3,   // ADVANCED_BOT_FACTORY
    4,   // SPECIAL_FACTORIES
    
    25,  // LIGHT_VEHICLES (scouts, raiders, light tanks)
    20,  // MEDIUM_VEHICLES (main battle tanks)
    15,  // HEAVY_VEHICLES (siege tanks, heavies)
    12,  // ARTILLERY_VEHICLES (mobile arty)
    8,   // AA_VEHICLES (mobile AA)
    10,  // SUPPORT_VEHICLES (radar, jammer, repair)
    
    15,  // FIGHTERS (interceptors, air superiority)
    12,  // BOMBERS (strategic, tactical, dive)
    8,   // GUNSHIPS (attack helicopters)
    6,   // TRANSPORT_AIRCRAFT
    8,   // RECONNAISSANCE_AIRCRAFT
    
    20,  // LIGHT_SHIPS (destroyers, subs, patrol)
    12,  // HEAVY_SHIPS (battleships, cruisers)
    4,   // AIRCRAFT_CARRIERS
    8,   // SUPPORT_SHIPS
    
    20,  // LIGHT_BOTS (infantry, scouts)
    15,  // MEDIUM_BOTS (assault infantry)
    10,  // HEAVY_BOTS (heavy assault)
    8,   // SPECIAL_BOTS
    
    9,   // BASIC_CONSTRUCTION (con units)
    6,   // ADVANCED_CONSTRUCTION (adv con)
    8,   // TRANSPORT_UTILITY
    6,   // REPAIR_UTILITY
    4,   // RECLAMATION_UTILITY
    
    3,   // COMMANDERS
    10,  // EXPERIMENTAL
    5,   // CAMPAIGN_SPECIAL
    20   // DECORATIVE
};

/**
 * Calculate total maximum units across all categories
 */
inline int GetTotalMaxUnits() {
    int total = 0;
    for (int i = 0; i < BUILD_CATEGORY_COUNT; ++i) {
        total += MAX_UNITS_PER_CATEGORY[i];
    }
    return total; // Should be ~400-500 total units
}

/**
 * Enhanced MLOutput with Hierarchical Build Selection
 */
struct MLOutput {
    // Existing heads
    std::vector<float> unitSelectionScores;           // [max_units]
    std::vector<std::vector<float>> spatialCommands;  // [11][height*width]
    std::vector<float> globalParams;                  // [6]
    float queueCommands;
    float urgentCommands;
    
    // HIERARCHICAL BUILD SELECTION - NEW
    // Stage 1: Category Selection (where to build what type of thing)
    std::vector<std::vector<float>> buildCategorySelection; // [50][height*width]
    
    // Stage 2: Specific Unit Selection (which specific unit in each category)
    std::vector<std::vector<float>> buildUnitInCategory;    // [category][max_units_in_category]
    
    int frame;
    
    MLOutput(int maxUnits = 2000, int mapWidth = 512, int mapHeight = 512) {
        unitSelectionScores.resize(maxUnits, 0.0f);
        spatialCommands.resize(SPATIAL_COMMAND_COUNT);
        for (auto& cmdMap : spatialCommands) {
            cmdMap.resize(mapWidth * mapHeight, 0.0f);
        }
        globalParams.resize(GLOBAL_PARAM_COUNT, 0.0f);
        
        // Initialize hierarchical build selection
        buildCategorySelection.resize(BUILD_CATEGORY_COUNT);
        for (auto& categoryMap : buildCategorySelection) {
            categoryMap.resize(mapWidth * mapHeight, 0.0f);
        }
        
        buildUnitInCategory.resize(BUILD_CATEGORY_COUNT);
        for (int i = 0; i < BUILD_CATEGORY_COUNT; ++i) {
            buildUnitInCategory[i].resize(MAX_UNITS_PER_CATEGORY[i], 0.0f);
        }
        
        queueCommands = 0.0f;
        urgentCommands = 0.0f;
        frame = 0;
    }
};

/**
 * Neural Network Architecture with Hierarchical Build Selection
 */
struct HierarchicalNetworkArchitecture {
    /*
    Input: [1, 968, 512, 512] (spatial game state)
        ↓ ConvNet Backbone (ResNet/EfficientNet) 
    Features: [1, 512, 64, 64]
        ↓ Split into specialized heads

    1. Unit Selection Head:
       Features → Global Average Pool → FC → [1, max_units]

    2. Spatial Command Head:  
       Features → Deconv/Upsample → [1, 11, 512, 512]

    3. Build Category Head (Stage 1):  
       Features → Deconv/Upsample → [1, 50, 512, 512]

    4. Build Unit Selection Head (Stage 2):
       Features → Global Average Pool → FC → [1, total_max_units_across_categories]
       Then reshape to [50][max_units_per_category]

    5. Global Parameter Head:
       Features → Global Average Pool → FC → [1, 6]

    6. Command Options Head:
       Features → Global Average Pool → FC → [1, 2]
    
    Total Output Size: ~105MB per inference
    - Unit selection: 2000 × 4 bytes = 8KB
    - Spatial commands: 11 × 512×512 × 4 bytes = 45MB  
    - Build categories: 50 × 512×512 × 4 bytes = 52MB
    - Build unit selection: 500 × 4 bytes = 2KB  
    - Global params: 6 × 4 bytes = 24 bytes
    - Command options: 2 × 4 bytes = 8 bytes
    */
};

/**
 * Hierarchical build command with resolved unit type
 */
struct HierarchicalBuildCommand {
    float x, y, z;
    BuildCategory category;
    int unitIndexInCategory;    // Index within the category
    int resolvedUnitDefId;      // Final Spring unit def ID
    std::string unitDefName;    // Unit definition name
    float categoryProbability;  // Probability from stage 1
    float unitProbability;      // Probability from stage 2
    float combinedProbability;  // categoryProbability × unitProbability
    int sourcePixelX, sourcePixelY;
    
    HierarchicalBuildCommand() : x(0), y(0), z(0), category(METAL_EXTRACTION),
                                unitIndexInCategory(0), resolvedUnitDefId(-1),
                                categoryProbability(0.0f), unitProbability(0.0f),
                                combinedProbability(0.0f), sourcePixelX(0), sourcePixelY(0) {}
};

/**
 * Category-to-UnitDef mapping for each faction
 */
class HierarchicalBuildMapper {
private:
    // Maps category + unit_index to actual unit def names for each faction
    std::unordered_map<std::string, std::vector<std::vector<std::string>>> factionBuildMaps;
    // Key: faction ("arm", "cor", "leg")
    // Value: [category][unit_index_in_category] -> unitDefName
    
    std::unordered_map<std::string, int> unitNameToDefId;
    std::string currentFaction;
    
public:
    HierarchicalBuildMapper();
    
    void InitializeFactionMappings();
    void SetCurrentFaction(const std::string& faction);
    
    // Resolve hierarchical selection to actual unit def
    int ResolveUnitDef(BuildCategory category, int unitIndex) const;
    std::string GetUnitDefName(BuildCategory category, int unitIndex) const;
    
    // Get all possible units in a category for current faction
    std::vector<std::string> GetUnitsInCategory(BuildCategory category) const;
    
    // Category suitability checks
    bool IsCategorySuitableForLocation(BuildCategory category, float x, float z) const;
    
private:
    void InitializeArmadaBuildMap();
    void InitializeCortexBuildMap(); 
    void InitializeLegionBuildMap();
};

} // namespace chanrts

#endif // _CHANRTS_HIERARCHICALBUILDSYSTEM_H