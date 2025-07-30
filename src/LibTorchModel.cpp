#include "LibTorchModel.h"
#include "ChanRTS.h"
#include "HierarchicalBuildSystem.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>

namespace chanrts {

// BackboneNetwork Implementation
BackboneNetwork::BackboneNetwork(const ModelConfig& config) : config(config) {
    if (config.backboneType == "resnet18" || config.backboneType == "resnet34" || config.backboneType == "resnet50") {
        BuildResNetBackbone();
    } else if (config.backboneType == "efficientnet") {
        BuildEfficientNetBackbone();
    } else if (config.backboneType == "custom") {
        BuildCustomBackbone();
    } else {
        throw std::runtime_error("Unknown backbone type: " + config.backboneType);
    }
    
    register_module("backbone", resnet_backbone);
    if (efficientnet_backbone) register_module("efficientnet", efficientnet_backbone);
    if (custom_backbone) register_module("custom", custom_backbone);
}

torch::Tensor BackboneNetwork::forward(torch::Tensor input) {
    // Input: [batch, 968, 512, 512]
    // Output: [batch, outputChannels, 64, 64] (assuming 8x reduction)
    
    if (resnet_backbone) {
        return resnet_backbone->forward(input);
    } else if (efficientnet_backbone) {
        return efficientnet_backbone->forward(input);
    } else if (custom_backbone) {
        return custom_backbone->forward(input);
    }
    
    throw std::runtime_error("No backbone network initialized");
}

void BackboneNetwork::BuildResNetBackbone() {
    spatialReduction = 8;  // 512 -> 64
    
    if (config.backboneType == "resnet18") {
        outputChannels = 512;
        
        resnet_backbone = torch::nn::Sequential(
            // Initial convolution - reduce channels from 968 to 64
            torch::nn::Conv2d(torch::nn::Conv2dOptions(config.inputChannels, 64, 7).stride(2).padding(3).bias(false)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)),
            
            // ResNet-18 blocks
            // Block 1: 64 channels
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)),
            torch::nn::BatchNorm2d(64),
            
            // Block 2: 128 channels
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1).bias(false)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(false)),
            torch::nn::BatchNorm2d(128),
            
            // Block 3: 256 channels
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(2).padding(1).bias(false)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)),
            torch::nn::BatchNorm2d(256),
            
            // Final block: 512 channels
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1).bias(false)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1).bias(false)),
            torch::nn::BatchNorm2d(512)
        );
        
    } else if (config.backboneType == "resnet34") {
        outputChannels = 512;
        // Similar structure but with more blocks
        // Implementation details would follow ResNet-34 architecture
        BuildResNet34();
        
    } else if (config.backboneType == "resnet50") {
        outputChannels = 2048;
        // ResNet-50 with bottleneck blocks
        BuildResNet50();
    }
}

void BackboneNetwork::BuildEfficientNetBackbone() {
    spatialReduction = 8;
    outputChannels = 512;
    
    // Simplified EfficientNet-like architecture
    efficientnet_backbone = torch::nn::Sequential(
        // Stem
        torch::nn::Conv2d(torch::nn::Conv2dOptions(config.inputChannels, 32, 3).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(32),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        
        // Mobile Inverted Bottleneck blocks
        // Stage 1
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        
        // Stage 2  
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(128),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        
        // Stage 3
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1).bias(false)),
        torch::nn::BatchNorm2d(256),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        
        // Head
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 1).stride(1).padding(0).bias(false)),
        torch::nn::BatchNorm2d(512),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
    );
}

void BackboneNetwork::BuildCustomBackbone() {
    spatialReduction = 8;
    outputChannels = 512;
    
    // Custom lightweight backbone optimized for RTS data
    custom_backbone = torch::nn::Sequential(
        // Channel reduction first
        torch::nn::Conv2d(torch::nn::Conv2dOptions(config.inputChannels, 128, 1).bias(false)),
        torch::nn::BatchNorm2d(128),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        
        // Spatial downsampling with depthwise separable convolutions
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(2).padding(1).groups(128).bias(false)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 1).bias(false)),
        torch::nn::BatchNorm2d(256),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(2).padding(1).groups(256).bias(false)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 1).bias(false)),
        torch::nn::BatchNorm2d(512),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        
        // Additional processing layers
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(512),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
    );
}

// MultiHeadNetwork Implementation
MultiHeadNetwork::MultiHeadNetwork(const ModelConfig& config, int inputChannels, int featureWidth, int featureHeight)
    : config(config), inputChannels(inputChannels), featureWidth(featureWidth), featureHeight(featureHeight) {
    
    BuildUnitSelectionHead();
    BuildSpatialCommandHead();
    BuildBuildCategoryHead();
    BuildBuildUnitHead();
    BuildGlobalParamHead();
    BuildCommandOptionHead();
    
    // Register all modules
    register_module("unit_selection", unitSelectionHead);
    register_module("spatial_command", spatialCommandHead);
    register_module("build_category", buildCategoryHead);
    register_module("build_unit", buildUnitHead);
    register_module("global_param", globalParamHead);
    register_module("command_option", commandOptionHead);
}

MultiHeadNetwork::MultiHeadOutput MultiHeadNetwork::forward(torch::Tensor features) {
    // features: [batch, channels, height, width]
    
    MultiHeadOutput output;
    
    // Unit selection: Global pooling -> FC
    auto pooled = torch::adaptive_avg_pool2d(features, {1, 1}).flatten(1);
    output.unitSelection = unitSelectionHead->forward(pooled);
    
    // Spatial commands: Upsample back to input resolution
    output.spatialCommands = spatialCommandHead->forward(features);
    
    // Build categories: Upsample back to input resolution
    output.buildCategories = buildCategoryHead->forward(features);
    
    // Build unit selection: Global pooling -> FC
    output.buildUnitSelection = buildUnitHead->forward(pooled);
    
    // Global parameters: Global pooling -> FC
    output.globalParams = globalParamHead->forward(pooled);
    
    // Command options: Global pooling -> FC
    output.commandOptions = commandOptionHead->forward(pooled);
    
    return output;
}

void MultiHeadNetwork::BuildUnitSelectionHead() {
    int pooledSize = inputChannels;  // After global average pooling
    
    unitSelectionHead = torch::nn::Sequential(
        torch::nn::Linear(pooledSize, 1024),
        torch::nn::ReLU(),
        torch::nn::Dropout(config.dropoutRate),
        torch::nn::Linear(1024, 512),
        torch::nn::ReLU(), 
        torch::nn::Linear(512, config.maxUnits),
        torch::nn::Sigmoid()  // Probabilities for unit selection
    );
}

void MultiHeadNetwork::BuildSpatialCommandHead() {
    // Upsample from feature resolution back to input resolution
    spatialCommandHead = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(inputChannels, 256, 3).padding(1)),
        torch::nn::BatchNorm2d(256),
        torch::nn::ReLU(),
        
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1)), // 2x upsample
        torch::nn::BatchNorm2d(128),
        torch::nn::ReLU(),
        
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1)),  // 4x upsample
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
        
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 32, 4).stride(2).padding(1)),   // 8x upsample
        torch::nn::BatchNorm2d(32),
        torch::nn::ReLU(),
        
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, config.spatialCommandTypes, 1)),
        torch::nn::Sigmoid()  // Probability maps for each command type
    );
}

void MultiHeadNetwork::BuildBuildCategoryHead() {
    // Similar to spatial command head but for build categories
    buildCategoryHead = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(inputChannels, 256, 3).padding(1)),
        torch::nn::BatchNorm2d(256),
        torch::nn::ReLU(),
        
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1)),
        torch::nn::BatchNorm2d(128),
        torch::nn::ReLU(),
        
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
        
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 32, 4).stride(2).padding(1)),
        torch::nn::BatchNorm2d(32),
        torch::nn::ReLU(),
        
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, config.buildCategories, 1)),
        torch::nn::Sigmoid()  // Probability maps for each build category
    );
}

void MultiHeadNetwork::BuildBuildUnitHead() {
    int pooledSize = inputChannels;
    int totalUnits = GetTotalMaxUnits();  // From HierarchicalBuildSystem.h
    
    buildUnitHead = torch::nn::Sequential(
        torch::nn::Linear(pooledSize, 512),
        torch::nn::ReLU(),
        torch::nn::Dropout(config.dropoutRate),
        torch::nn::Linear(512, 256),
        torch::nn::ReLU(),
        torch::nn::Linear(256, totalUnits),
        torch::nn::Softmax(torch::nn::SoftmaxOptions(1))  // Probability distribution over all units
    );
}

void MultiHeadNetwork::BuildGlobalParamHead() {
    int pooledSize = inputChannels;
    
    globalParamHead = torch::nn::Sequential(
        torch::nn::Linear(pooledSize, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, config.globalParams),
        torch::nn::Sigmoid()  // Global parameter values 0-1
    );
}

void MultiHeadNetwork::BuildCommandOptionHead() {
    int pooledSize = inputChannels;
    
    commandOptionHead = torch::nn::Sequential(
        torch::nn::Linear(pooledSize, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, config.commandOptions),
        torch::nn::Sigmoid()  // Command option probabilities
    );
}

// HierarchicalRTSModel Implementation
HierarchicalRTSModel::HierarchicalRTSModel(const ModelConfig& config)
    : config(config), device(torch::kCPU), isInitialized(false), isOnGPU(false) {
    
    // Initialize backbone
    backbone = std::make_unique<BackboneNetwork>(config);
    
    // Calculate feature dimensions after backbone
    int featureChannels = backbone->GetOutputChannels();
    int featureWidth = config.inputWidth / backbone->GetSpatialReduction();
    int featureHeight = config.inputHeight / backbone->GetSpatialReduction();
    
    // Initialize multi-head network
    heads = std::make_unique<MultiHeadNetwork>(config, featureChannels, featureWidth, featureHeight);
    
    // Register modules
    register_module("backbone", backbone);
    register_module("heads", heads);
    
    // Set device
    SetDevice(config.device);
    
    isInitialized = true;
}

HierarchicalRTSModel::~HierarchicalRTSModel() {
    ClearCache();
}

void HierarchicalRTSModel::LoadFromFile(const std::string& modelPath) {
    try {
        torch::load(*this, modelPath);
        std::cout << "Successfully loaded model from: " << modelPath << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model from " + modelPath + ": " + e.what());
    }
}

void HierarchicalRTSModel::SaveToFile(const std::string& modelPath) const {
    try {
        torch::save(*this, modelPath);
        std::cout << "Successfully saved model to: " << modelPath << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to save model to " + modelPath + ": " + e.what());
    }
}

void HierarchicalRTSModel::SetDevice(const std::string& deviceStr) {
    if (deviceStr == "cuda" && torch::cuda::is_available()) {
        device = torch::kCUDA;
        this->to(device);
        isOnGPU = true;
        std::cout << "Model moved to CUDA device" << std::endl;
    } else {
        device = torch::kCPU;
        this->to(device);
        isOnGPU = false;
        std::cout << "Model using CPU device" << std::endl;
    }
}

void HierarchicalRTSModel::SetEvalMode() {
    this->eval();
}

void HierarchicalRTSModel::SetTrainMode() {
    this->train();
}

MLOutput HierarchicalRTSModel::Inference(const SpatialTensor& input) {
    torch::NoGradGuard no_grad;  // Disable gradient computation for inference
    
    auto torchInput = SpatialTensorToTorch(input);
    ValidateInputDimensions(torchInput);
    
    // Forward pass through backbone
    auto features = backbone->forward(torchInput);
    
    // Forward pass through heads
    auto torchOutput = heads->forward(features);
    
    // Convert back to MLOutput
    return TorchToMLOutput(torchOutput);
}

std::vector<MLOutput> HierarchicalRTSModel::BatchInference(const std::vector<SpatialTensor>& inputs) {
    torch::NoGradGuard no_grad;
    
    // Convert batch to torch tensor
    std::vector<torch::Tensor> torchInputs;
    for (const auto& input : inputs) {
        torchInputs.push_back(SpatialTensorToTorch(input));
    }
    
    auto batchInput = torch::stack(torchInputs);
    ValidateInputDimensions(batchInput);
    
    // Forward pass
    auto features = backbone->forward(batchInput);
    auto torchOutput = heads->forward(features);
    
    // Convert batch results back to MLOutput vector
    std::vector<MLOutput> results;
    for (int i = 0; i < inputs.size(); ++i) {
        // Extract individual results from batch
        MultiHeadNetwork::MultiHeadOutput singleOutput;
        singleOutput.unitSelection = torchOutput.unitSelection[i];
        singleOutput.spatialCommands = torchOutput.spatialCommands[i];
        singleOutput.buildCategories = torchOutput.buildCategories[i];
        singleOutput.buildUnitSelection = torchOutput.buildUnitSelection[i];
        singleOutput.globalParams = torchOutput.globalParams[i];
        singleOutput.commandOptions = torchOutput.commandOptions[i];
        
        results.push_back(TorchToMLOutput(singleOutput));
    }
    
    return results;
}

torch::Tensor HierarchicalRTSModel::SpatialTensorToTorch(const SpatialTensor& spatial) {
    // Convert SpatialTensor to torch::Tensor
    // Input: spatial.layers[channel][height*width]
    // Output: [1, channels, height, width]
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto tensor = torch::zeros({1, SpatialTensor::CHANNELS, spatial.height, spatial.width}, options);
    
    for (int c = 0; c < SpatialTensor::CHANNELS; ++c) {
        for (int h = 0; h < spatial.height; ++h) {
            for (int w = 0; w < spatial.width; ++w) {
                int index = h * spatial.width + w;
                tensor[0][c][h][w] = spatial.layers[c][index];
            }
        }
    }
    
    return tensor;
}

MLOutput HierarchicalRTSModel::TorchToMLOutput(const MultiHeadNetwork::MultiHeadOutput& torchOutput) {
    MLOutput output(config.maxUnits, config.inputWidth, config.inputHeight);
    
    // Convert unit selection
    auto unitSelectionData = torchOutput.unitSelection.to(torch::kCPU);
    auto unitSelectionAccessor = unitSelectionData.accessor<float, 1>();
    for (int i = 0; i < config.maxUnits; ++i) {
        output.unitSelectionScores[i] = unitSelectionAccessor[i];
    }
    
    // Convert spatial commands
    auto spatialData = torchOutput.spatialCommands.to(torch::kCPU);
    auto spatialAccessor = spatialData.accessor<float, 3>();
    for (int cmd = 0; cmd < config.spatialCommandTypes; ++cmd) {
        for (int h = 0; h < config.inputHeight; ++h) {
            for (int w = 0; w < config.inputWidth; ++w) {
                int index = h * config.inputWidth + w;
                output.spatialCommands[cmd][index] = spatialAccessor[cmd][h][w];
            }
        }
    }
    
    // Convert build categories
    auto buildCatData = torchOutput.buildCategories.to(torch::kCPU);
    auto buildCatAccessor = buildCatData.accessor<float, 3>();
    for (int cat = 0; cat < config.buildCategories; ++cat) {
        for (int h = 0; h < config.inputHeight; ++h) {
            for (int w = 0; w < config.inputWidth; ++w) {
                int index = h * config.inputWidth + w;
                output.buildCategorySelection[cat][index] = buildCatAccessor[cat][h][w];
            }
        }
    }
    
    // Convert build unit selection
    auto buildUnitData = torchOutput.buildUnitSelection.to(torch::kCPU);
    auto buildUnitAccessor = buildUnitData.accessor<float, 1>();
    int unitIndex = 0;
    for (int cat = 0; cat < BUILD_CATEGORY_COUNT; ++cat) {
        for (int unit = 0; unit < MAX_UNITS_PER_CATEGORY[cat]; ++unit) {
            if (unitIndex < buildUnitAccessor.size(0)) {
                output.buildUnitInCategory[cat][unit] = buildUnitAccessor[unitIndex];
                unitIndex++;
            }
        }
    }
    
    // Convert global parameters
    auto globalData = torchOutput.globalParams.to(torch::kCPU);
    auto globalAccessor = globalData.accessor<float, 1>();
    for (int i = 0; i < config.globalParams; ++i) {
        output.globalParams[i] = globalAccessor[i];
    }
    
    // Convert command options
    auto optionData = torchOutput.commandOptions.to(torch::kCPU);
    auto optionAccessor = optionData.accessor<float, 1>();
    output.queueCommands = optionAccessor[0];
    output.urgentCommands = optionAccessor[1];
    
    return output;
}

void HierarchicalRTSModel::ValidateInputDimensions(const torch::Tensor& input) {
    auto sizes = input.sizes();
    if (sizes.size() != 4) {
        throw std::runtime_error("Input tensor must be 4D [batch, channels, height, width]");
    }
    if (sizes[1] != config.inputChannels) {
        throw std::runtime_error("Input channels mismatch: expected " + std::to_string(config.inputChannels) + 
                                ", got " + std::to_string(sizes[1]));
    }
    if (sizes[2] != config.inputHeight || sizes[3] != config.inputWidth) {
        throw std::runtime_error("Input spatial dimensions mismatch: expected " + 
                                std::to_string(config.inputHeight) + "x" + std::to_string(config.inputWidth) +
                                ", got " + std::to_string(sizes[2]) + "x" + std::to_string(sizes[3]));
    }
}

size_t HierarchicalRTSModel::GetParameterCount() const {
    size_t count = 0;
    for (const auto& param : parameters()) {
        count += param.numel();
    }
    return count;
}

size_t HierarchicalRTSModel::GetMemoryUsage() const {
    size_t memory = 0;
    for (const auto& param : parameters()) {
        memory += param.numel() * sizeof(float);
    }
    return memory;
}

void HierarchicalRTSModel::PrintModelSummary() const {
    std::cout << "\n=== Hierarchical RTS Model Summary ===" << std::endl;
    std::cout << "Backbone: " << config.backboneType << std::endl;
    std::cout << "Input: [" << config.inputChannels << ", " << config.inputHeight << ", " << config.inputWidth << "]" << std::endl;
    std::cout << "Parameters: " << GetParameterCount() << std::endl;
    std::cout << "Memory Usage: " << GetMemoryUsage() / (1024*1024) << " MB" << std::endl;
    std::cout << "Device: " << (isOnGPU ? "CUDA" : "CPU") << std::endl;
    std::cout << "======================================\n" << std::endl;
}

void HierarchicalRTSModel::OptimizeMemoryUsage() {
    // Clear any cached tensors
    ClearCache();
    
    // Force garbage collection
    if (isOnGPU) {
        torch::cuda::empty_cache();
    }
}

void HierarchicalRTSModel::ClearCache() {
    if (isOnGPU) {
        torch::cuda::empty_cache();
    }
}

// ModelFactory Implementation
ModelConfig ModelFactory::GetResNet18Config() {
    ModelConfig config;
    config.backboneType = "resnet18";
    config.backboneFeatures = 512;
    config.useDropout = true;
    config.dropoutRate = 0.1f;
    return config;
}

ModelConfig ModelFactory::GetResNet34Config() {
    ModelConfig config;
    config.backboneType = "resnet34";
    config.backboneFeatures = 512;
    config.useDropout = true;
    config.dropoutRate = 0.15f;
    return config;
}

ModelConfig ModelFactory::GetCustomLightweightConfig() {
    ModelConfig config;
    config.backboneType = "custom";
    config.backboneFeatures = 256;
    config.useDropout = false;
    config.useBatchNorm = true;
    return config;
}

std::unique_ptr<HierarchicalRTSModel> ModelFactory::CreateModel(const ModelConfig& config) {
    if (!ValidateConfig(config)) {
        throw std::runtime_error("Invalid model configuration");
    }
    
    return std::make_unique<HierarchicalRTSModel>(config);
}

bool ModelFactory::ValidateConfig(const ModelConfig& config) {
    if (config.inputChannels != 968) return false;
    if (config.inputWidth <= 0 || config.inputHeight <= 0) return false;
    if (config.maxUnits <= 0) return false;
    if (config.backboneType.empty()) return false;
    return true;
}

} // namespace chanrts