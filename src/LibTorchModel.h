#ifndef _CHANRTS_LIBTORCHMODEL_H
#define _CHANRTS_LIBTORCHMODEL_H

#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace chanrts {

// Forward declarations
struct SpatialTensor;
struct MLOutput;

/**
 * Model architecture configuration
 */
struct ModelConfig {
    // Input dimensions
    int inputChannels = 968;
    int inputWidth = 512;
    int inputHeight = 512;
    
    // Backbone architecture
    std::string backboneType = "resnet34";  // "resnet18", "resnet34", "resnet50", "efficientnet", "custom"
    int backboneFeatures = 512;             // Feature dimensions from backbone
    int backboneReduction = 8;              // Spatial reduction factor (input/output)
    
    // Output head configurations
    int maxUnits = 2000;
    int spatialCommandTypes = 11;
    int buildCategories = 50;
    int maxUnitsPerCategory = 25;           // Max across all categories
    int globalParams = 6;
    int commandOptions = 2;
    
    // Training parameters
    bool useDropout = true;
    float dropoutRate = 0.1f;
    bool useBatchNorm = true;
    bool useAttention = false;              // Optional attention mechanisms
    
    // Hardware configuration  
    std::string device = "cuda";            // "cuda", "cpu"
    bool useHalfPrecision = false;          // FP16 for faster inference
    int batchSize = 1;                      // For batch inference
    
    ModelConfig() = default;
    
    // Load from JSON configuration file
    static ModelConfig FromFile(const std::string& configPath);
    void SaveToFile(const std::string& configPath) const;
};

/**
 * Flexible backbone networks
 */
class BackboneNetwork : public torch::nn::Module {
public:
    BackboneNetwork(const ModelConfig& config);
    torch::Tensor forward(torch::Tensor input);
    
    int GetOutputChannels() const { return outputChannels; }
    int GetSpatialReduction() const { return spatialReduction; }
    
private:
    ModelConfig config;
    int outputChannels;
    int spatialReduction;
    
    // Different backbone implementations
    torch::nn::Sequential resnet_backbone{nullptr};
    torch::nn::Sequential efficientnet_backbone{nullptr};
    torch::nn::Sequential custom_backbone{nullptr};
    
    void BuildResNetBackbone();
    void BuildEfficientNetBackbone();
    void BuildCustomBackbone();
};

/**
 * Multi-head output network
 */
class MultiHeadNetwork : public torch::nn::Module {
public:
    MultiHeadNetwork(const ModelConfig& config, int inputChannels, int featureWidth, int featureHeight);
    
    struct MultiHeadOutput {
        torch::Tensor unitSelection;        // [batch, maxUnits]
        torch::Tensor spatialCommands;      // [batch, commandTypes, height, width]
        torch::Tensor buildCategories;      // [batch, categories, height, width] 
        torch::Tensor buildUnitSelection;   // [batch, totalUnitsAcrossCategories]
        torch::Tensor globalParams;        // [batch, globalParams]
        torch::Tensor commandOptions;      // [batch, commandOptions]
    };
    
    MultiHeadOutput forward(torch::Tensor features);
    
private:
    ModelConfig config;
    int inputChannels;
    int featureWidth, featureHeight;
    
    // Individual heads
    torch::nn::Sequential unitSelectionHead{nullptr};
    torch::nn::Sequential spatialCommandHead{nullptr};
    torch::nn::Sequential buildCategoryHead{nullptr};
    torch::nn::Sequential buildUnitHead{nullptr};
    torch::nn::Sequential globalParamHead{nullptr};
    torch::nn::Sequential commandOptionHead{nullptr};
    
    void BuildUnitSelectionHead();
    void BuildSpatialCommandHead();
    void BuildBuildCategoryHead();
    void BuildBuildUnitHead();
    void BuildGlobalParamHead();
    void BuildCommandOptionHead();
};

/**
 * Complete hierarchical RTS model
 */
class HierarchicalRTSModel : public torch::nn::Module {
public:
    HierarchicalRTSModel(const ModelConfig& config);
    ~HierarchicalRTSModel();
    
    // Model operations
    void LoadFromFile(const std::string& modelPath);
    void SaveToFile(const std::string& modelPath) const;
    void SetDevice(const std::string& device);
    void SetEvalMode();
    void SetTrainMode();
    
    // Inference
    MLOutput Inference(const SpatialTensor& input);
    std::vector<MLOutput> BatchInference(const std::vector<SpatialTensor>& inputs);
    
    // Training support
    struct TrainingBatch {
        torch::Tensor inputs;               // [batch, channels, height, width]
        torch::Tensor unitSelectionTargets;
        torch::Tensor spatialCommandTargets;
        torch::Tensor buildCategoryTargets;
        torch::Tensor buildUnitTargets;
        torch::Tensor globalParamTargets;
        torch::Tensor commandOptionTargets;
    };
    
    struct ModelLoss {
        torch::Tensor totalLoss;
        torch::Tensor unitSelectionLoss;
        torch::Tensor spatialCommandLoss;
        torch::Tensor buildCategoryLoss;
        torch::Tensor buildUnitLoss;
        torch::Tensor globalParamLoss;
        torch::Tensor commandOptionLoss;
    };
    
    ModelLoss ComputeLoss(const TrainingBatch& batch);
    
    // Model introspection
    ModelConfig GetConfig() const { return config; }
    size_t GetParameterCount() const;
    size_t GetMemoryUsage() const;
    void PrintModelSummary() const;
    
private:
    ModelConfig config;
    torch::Device device;
    
    std::unique_ptr<BackboneNetwork> backbone;
    std::unique_ptr<MultiHeadNetwork> heads;
    
    // Model state
    bool isInitialized;
    bool isOnGPU;
    
    // Conversion utilities
    torch::Tensor SpatialTensorToTorch(const SpatialTensor& spatial);
    MLOutput TorchToMLOutput(const MultiHeadNetwork::MultiHeadOutput& torchOutput);
    void ValidateInputDimensions(const torch::Tensor& input);
    
    // Memory management
    void OptimizeMemoryUsage();
    void ClearCache();
};

/**
 * Model factory for creating different architectures
 */
class ModelFactory {
public:
    // Predefined model configurations
    static ModelConfig GetResNet18Config();
    static ModelConfig GetResNet34Config();
    static ModelConfig GetResNet50Config();
    static ModelConfig GetEfficientNetB0Config();
    static ModelConfig GetEfficientNetB4Config();
    static ModelConfig GetCustomLightweightConfig();
    static ModelConfig GetCustomHighCapacityConfig();
    
    // Model creation
    static std::unique_ptr<HierarchicalRTSModel> CreateModel(const ModelConfig& config);
    static std::unique_ptr<HierarchicalRTSModel> CreateModelFromFile(const std::string& configPath);
    
    // Architecture validation
    static bool ValidateConfig(const ModelConfig& config);
    static std::string GetConfigSummary(const ModelConfig& config);
};

/**
 * Inference optimization utilities
 */
class InferenceOptimizer {
public:
    static void OptimizeForInference(HierarchicalRTSModel& model);
    static void EnableTensorRT(HierarchicalRTSModel& model);  // If available
    static void EnableQuantization(HierarchicalRTSModel& model);
    static void WarmupModel(HierarchicalRTSModel& model, const ModelConfig& config);
    
    // Performance benchmarking
    struct BenchmarkResults {
        double avgInferenceTime;     // milliseconds
        double minInferenceTime;
        double maxInferenceTime;
        size_t memoryUsage;          // bytes
        double throughput;           // inferences per second
    };
    
    static BenchmarkResults BenchmarkModel(HierarchicalRTSModel& model, 
                                          const ModelConfig& config,
                                          int iterations = 100);
};

/**
 * Model training utilities (for future use)
 */
class ModelTrainer {
public:
    struct TrainingConfig {
        float learningRate = 0.001f;
        int batchSize = 4;
        int epochs = 100;
        std::string optimizer = "adam";     // "adam", "sgd", "adamw"
        float weightDecay = 0.0001f;
        bool useLRScheduler = true;
        std::string lrScheduler = "cosine"; // "cosine", "step", "exponential"
        
        // Loss weights for multi-head training
        float unitSelectionWeight = 1.0f;
        float spatialCommandWeight = 2.0f;
        float buildCategoryWeight = 2.0f;
        float buildUnitWeight = 1.0f;
        float globalParamWeight = 0.5f;
        float commandOptionWeight = 0.5f;
    };
    
    ModelTrainer(std::unique_ptr<HierarchicalRTSModel> model, const TrainingConfig& config);
    
    void Train(const std::vector<HierarchicalRTSModel::TrainingBatch>& trainingData);
    void Validate(const std::vector<HierarchicalRTSModel::TrainingBatch>& validationData);
    void SaveCheckpoint(const std::string& path, int epoch);
    void LoadCheckpoint(const std::string& path);
    
private:
    std::unique_ptr<HierarchicalRTSModel> model;
    TrainingConfig config;
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    std::unique_ptr<torch::optim::LRScheduler> scheduler;
};

} // namespace chanrts

#endif // _CHANRTS_LIBTORCHMODEL_H