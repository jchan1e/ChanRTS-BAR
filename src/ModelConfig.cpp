#include "ModelConfig.h"
#include "LibTorchModel.h"

#include <fstream>
#include <iostream>
#include <sstream>

// Simple JSON-like parser (basic implementation)
// For production, consider using a proper JSON library like nlohmann/json

namespace chanrts {

std::unordered_map<std::string, std::string> ModelConfigManager::backboneTemplates;

ModelConfig ModelConfigManager::LoadFromJSON(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filepath);
    }
    
    ModelConfig config;
    std::string line;
    
    // Simple key-value parser (replace with proper JSON library in production)
    while (std::getline(file, line)) {
        if (line.find("\"backbone_type\"") != std::string::npos) {
            size_t start = line.find(":") + 1;
            size_t begin = line.find("\"", start) + 1;
            size_t end = line.find("\"", begin);
            config.backboneType = line.substr(begin, end - begin);
        }
        else if (line.find("\"input_channels\"") != std::string::npos) {
            size_t start = line.find(":") + 1;
            config.inputChannels = std::stoi(line.substr(start));
        }
        else if (line.find("\"input_width\"") != std::string::npos) {
            size_t start = line.find(":") + 1;
            config.inputWidth = std::stoi(line.substr(start));
        }
        else if (line.find("\"input_height\"") != std::string::npos) {
            size_t start = line.find(":") + 1;
            config.inputHeight = std::stoi(line.substr(start));
        }
        else if (line.find("\"backbone_features\"") != std::string::npos) {
            size_t start = line.find(":") + 1;
            config.backboneFeatures = std::stoi(line.substr(start));
        }
        else if (line.find("\"max_units\"") != std::string::npos) {
            size_t start = line.find(":") + 1;
            config.maxUnits = std::stoi(line.substr(start));
        }
        else if (line.find("\"use_dropout\"") != std::string::npos) {
            config.useDropout = line.find("true") != std::string::npos;
        }
        else if (line.find("\"dropout_rate\"") != std::string::npos) {
            size_t start = line.find(":") + 1;
            config.dropoutRate = std::stof(line.substr(start));
        }
        else if (line.find("\"device\"") != std::string::npos) {
            size_t start = line.find(":") + 1;
            size_t begin = line.find("\"", start) + 1;
            size_t end = line.find("\"", begin);
            config.device = line.substr(begin, end - begin);
        }
        else if (line.find("\"use_half_precision\"") != std::string::npos) {
            config.useHalfPrecision = line.find("true") != std::string::npos;
        }
    }
    
    return config;
}

void ModelConfigManager::SaveToJSON(const ModelConfig& config, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create config file: " + filepath);
    }
    
    file << "{\n";
    file << "  \"backbone_type\": \"" << config.backboneType << "\",\n";
    file << "  \"input_channels\": " << config.inputChannels << ",\n";
    file << "  \"input_width\": " << config.inputWidth << ",\n";
    file << "  \"input_height\": " << config.inputHeight << ",\n";
    file << "  \"backbone_features\": " << config.backboneFeatures << ",\n";
    file << "  \"backbone_reduction\": " << config.backboneReduction << ",\n";
    file << "  \"max_units\": " << config.maxUnits << ",\n";
    file << "  \"spatial_command_types\": " << config.spatialCommandTypes << ",\n";
    file << "  \"build_categories\": " << config.buildCategories << ",\n";
    file << "  \"max_units_per_category\": " << config.maxUnitsPerCategory << ",\n";
    file << "  \"global_params\": " << config.globalParams << ",\n";
    file << "  \"command_options\": " << config.commandOptions << ",\n";
    file << "  \"use_dropout\": " << (config.useDropout ? "true" : "false") << ",\n";
    file << "  \"dropout_rate\": " << config.dropoutRate << ",\n";
    file << "  \"use_batch_norm\": " << (config.useBatchNorm ? "true" : "false") << ",\n";
    file << "  \"use_attention\": " << (config.useAttention ? "true" : "false") << ",\n";
    file << "  \"device\": \"" << config.device << "\",\n";
    file << "  \"use_half_precision\": " << (config.useHalfPrecision ? "true" : "false") << ",\n";
    file << "  \"batch_size\": " << config.batchSize << "\n";
    file << "}\n";
    
    file.close();
}

void ModelConfigManager::CreateDefaultConfigs(const std::string& configDir) {
    // Create default configurations for different use cases
    
    // ResNet-18 for fast inference
    auto resnet18Config = ModelFactory::GetResNet18Config();
    SaveToJSON(resnet18Config, configDir + "/resnet18_fast.json");
    
    // ResNet-34 for balanced performance
    auto resnet34Config = ModelFactory::GetResNet34Config();
    SaveToJSON(resnet34Config, configDir + "/resnet34_balanced.json");
    
    // Custom lightweight for low-end hardware
    auto lightweightConfig = ModelFactory::GetCustomLightweightConfig();
    SaveToJSON(lightweightConfig, configDir + "/custom_lightweight.json");
    
    // High-capacity configuration for training
    ModelConfig highCapacityConfig;
    highCapacityConfig.backboneType = "resnet50";
    highCapacityConfig.backboneFeatures = 2048;
    highCapacityConfig.useDropout = true;
    highCapacityConfig.dropoutRate = 0.2f;
    highCapacityConfig.useAttention = true;
    SaveToJSON(highCapacityConfig, configDir + "/resnet50_high_capacity.json");
    
    // CPU-optimized configuration
    ModelConfig cpuConfig = ModelFactory::GetCustomLightweightConfig();
    cpuConfig.device = "cpu";
    cpuConfig.batchSize = 1;
    cpuConfig.useHalfPrecision = false;
    SaveToJSON(cpuConfig, configDir + "/cpu_optimized.json");
    
    std::cout << "Created default configurations in: " << configDir << std::endl;
}

bool ModelConfigManager::ValidateConfiguration(const ModelConfig& config) {
    // Input validation
    if (config.inputChannels != 968) return false;
    if (config.inputWidth <= 0 || config.inputHeight <= 0) return false;
    if (config.inputWidth % 32 != 0 || config.inputHeight % 32 != 0) return false;  // Must be divisible by 32
    
    // Architecture validation
    if (config.backboneType.empty()) return false;
    if (config.backboneFeatures <= 0) return false;
    if (config.backboneReduction <= 0) return false;
    
    // Output validation
    if (config.maxUnits <= 0 || config.maxUnits > 10000) return false;
    if (config.spatialCommandTypes != 11) return false;
    if (config.buildCategories != 50) return false;
    
    // Training parameter validation
    if (config.dropoutRate < 0.0f || config.dropoutRate > 1.0f) return false;
    if (config.batchSize <= 0 || config.batchSize > 128) return false;
    
    return true;
}

std::string ModelConfigManager::GetValidationErrors(const ModelConfig& config) {
    std::stringstream errors;
    
    if (config.inputChannels != 968) {
        errors << "Input channels must be 968, got " << config.inputChannels << "\n";
    }
    if (config.inputWidth <= 0 || config.inputHeight <= 0) {
        errors << "Input dimensions must be positive\n";
    }
    if (config.inputWidth % 32 != 0 || config.inputHeight % 32 != 0) {
        errors << "Input dimensions must be divisible by 32\n";
    }
    if (config.backboneType.empty()) {
        errors << "Backbone type cannot be empty\n";
    }
    if (config.maxUnits <= 0 || config.maxUnits > 10000) {
        errors << "Max units must be between 1 and 10000\n";
    }
    if (config.dropoutRate < 0.0f || config.dropoutRate > 1.0f) {
        errors << "Dropout rate must be between 0.0 and 1.0\n";
    }
    
    return errors.str();
}

ModelConfig ModelConfigManager::CreateResNetConfig(int layers, bool useAttention) {
    ModelConfig config;
    
    switch (layers) {
        case 18:
            config.backboneType = "resnet18";
            config.backboneFeatures = 512;
            break;
        case 34:
            config.backboneType = "resnet34";
            config.backboneFeatures = 512;
            break;
        case 50:
            config.backboneType = "resnet50";
            config.backboneFeatures = 2048;
            break;
        default:
            throw std::runtime_error("Unsupported ResNet layer count: " + std::to_string(layers));
    }
    
    config.useAttention = useAttention;
    config.useDropout = true;
    config.dropoutRate = layers >= 50 ? 0.2f : 0.1f;
    
    return config;
}

ModelConfig ModelConfigManager::CreateEfficientNetConfig(const std::string& variant) {
    ModelConfig config;
    config.backboneType = "efficientnet";
    
    if (variant == "b0") {
        config.backboneFeatures = 320;
        config.dropoutRate = 0.2f;
    } else if (variant == "b4") {
        config.backboneFeatures = 448;
        config.dropoutRate = 0.4f;
    } else {
        throw std::runtime_error("Unsupported EfficientNet variant: " + variant);
    }
    
    config.useDropout = true;
    config.useBatchNorm = true;
    
    return config;
}

ModelConfig ModelConfigManager::CreateCustomConfig(const std::string& name) {
    ModelConfig config;
    config.backboneType = "custom";
    
    if (name == "lightweight") {
        config.backboneFeatures = 256;
        config.useDropout = false;
        config.useBatchNorm = true;
    } else if (name == "high_capacity") {
        config.backboneFeatures = 1024;
        config.useDropout = true;
        config.dropoutRate = 0.25f;
        config.useAttention = true;
    } else {
        throw std::runtime_error("Unknown custom configuration: " + name);
    }
    
    return config;
}

// DeploymentConfig implementations
DeploymentConfig DeploymentConfig::ForRealTimeInference() {
    DeploymentConfig config;
    config.enableOptimizations = true;
    config.useQuantization = true;
    config.maxBatchSize = 1;
    config.numThreads = 2;
    config.inferenceTimeoutMs = 16.0f;  // 60 FPS target
    config.enableAsyncInference = true;
    config.queueSize = 2;
    config.maxMemoryUsage = 1ULL * 1024 * 1024 * 1024;  // 1GB
    return config;
}

DeploymentConfig DeploymentConfig::ForBatchProcessing() {
    DeploymentConfig config;
    config.enableOptimizations = true;
    config.useQuantization = false;
    config.maxBatchSize = 8;
    config.numThreads = 8;
    config.inferenceTimeoutMs = 1000.0f;  // More relaxed timing
    config.enableAsyncInference = false;
    config.maxMemoryUsage = 8ULL * 1024 * 1024 * 1024;  // 8GB
    return config;
}

DeploymentConfig DeploymentConfig::ForLowMemoryDevice() {
    DeploymentConfig config;
    config.enableOptimizations = true;
    config.useQuantization = true;
    config.maxBatchSize = 1;
    config.numThreads = 1;
    config.inferenceTimeoutMs = 100.0f;
    config.enableAsyncInference = false;
    config.maxMemoryUsage = 512ULL * 1024 * 1024;  // 512MB
    config.useMemoryMapping = true;
    return config;
}

} // namespace chanrts