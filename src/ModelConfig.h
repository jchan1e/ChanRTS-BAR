#ifndef _CHANRTS_MODELCONFIG_H
#define _CHANRTS_MODELCONFIG_H

#include <string>
#include <unordered_map>
#include <vector>
#include "LibTorchModel.h"

namespace chanrts {

/**
 * JSON configuration file format for model architectures
 */
class ModelConfigManager {
public:
    // Load/Save configuration from JSON files
    static ModelConfig LoadFromJSON(const std::string& filepath);
    static void SaveToJSON(const ModelConfig& config, const std::string& filepath);
    
    // Predefined configurations  
    static void CreateDefaultConfigs(const std::string& configDir);
    
    // Configuration validation
    static bool ValidateConfiguration(const ModelConfig& config);
    static std::string GetValidationErrors(const ModelConfig& config);
    
    // Architecture templates
    static ModelConfig CreateResNetConfig(int layers, bool useAttention = false);
    static ModelConfig CreateEfficientNetConfig(const std::string& variant);
    static ModelConfig CreateCustomConfig(const std::string& name);
    
private:
    static std::unordered_map<std::string, std::string> backboneTemplates;
    static void InitializeTemplates();
};

/**
 * Runtime model configuration for different deployment scenarios
 */
struct DeploymentConfig {
    // Performance settings
    bool enableOptimizations = true;
    bool useQuantization = false;
    bool useTensorRT = false;
    int maxBatchSize = 1;
    int numThreads = 4;
    
    // Memory settings
    bool enableMemoryPool = true;
    size_t maxMemoryUsage = 2ULL * 1024 * 1024 * 1024;  // 2GB
    bool useMemoryMapping = false;
    
    // Inference settings
    float inferenceTimeoutMs = 100.0f;
    bool enableAsyncInference = true;
    int queueSize = 4;
    
    // Fallback settings
    bool enableFallback = true;
    std::string fallbackModelPath = "";
    float fallbackThreshold = 0.95f;  // Use fallback if confidence < threshold
    
    DeploymentConfig() = default;
    
    static DeploymentConfig ForRealTimeInference();
    static DeploymentConfig ForBatchProcessing();
    static DeploymentConfig ForLowMemoryDevice();
};

} // namespace chanrts

#endif // _CHANRTS_MODELCONFIG_H