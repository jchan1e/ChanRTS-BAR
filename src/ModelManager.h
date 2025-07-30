#ifndef _CHANRTS_MODELMANAGER_H
#define _CHANRTS_MODELMANAGER_H

#include "LibTorchModel.h"
#include "ModelConfig.h"

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <queue>
#include <condition_variable>

namespace chanrts {

// Forward declarations
struct SpatialTensor;
struct MLOutput;

/**
 * High-level model management for ChanRTS AI
 */
class ModelManager {
public:
    ModelManager();
    ~ModelManager();
    
    // Model lifecycle management
    bool LoadModel(const std::string& modelPath, const std::string& configPath = "");
    bool LoadModelFromConfig(const ModelConfig& config, const std::string& weightsPath = "");
    void UnloadModel();
    bool IsModelLoaded() const { return model != nullptr; }
    
    // Model selection and switching
    bool HasMultipleModels() const { return modelVariants.size() > 1; }
    bool SwitchToModel(const std::string& modelName);
    std::vector<std::string> GetAvailableModels() const;
    std::string GetCurrentModelName() const { return currentModelName; }
    
    // Inference interface
    MLOutput RunInference(const SpatialTensor& input);
    std::vector<MLOutput> RunBatchInference(const std::vector<SpatialTensor>& inputs);
    
    // Asynchronous inference
    struct InferenceRequest {
        SpatialTensor input;
        int requestId;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    struct InferenceResult {
        MLOutput output;
        int requestId;
        bool success;
        std::string errorMessage;
        double inferenceTimeMs;
    };
    
    int SubmitAsyncInference(const SpatialTensor& input);
    bool GetInferenceResult(int requestId, InferenceResult& result);
    void CancelInference(int requestId);
    
    // Performance monitoring
    struct PerformanceStats {
        double avgInferenceTime;
        double minInferenceTime;
        double maxInferenceTime;
        size_t totalInferences;
        size_t failedInferences;
        double throughput;          // inferences per second
        size_t memoryUsage;
        std::chrono::steady_clock::time_point lastUpdate;
    };
    
    PerformanceStats GetPerformanceStats() const;
    void ResetPerformanceStats();
    
    // Model introspection
    ModelConfig GetCurrentModelConfig() const;
    size_t GetModelParameterCount() const;
    size_t GetModelMemoryUsage() const;
    void PrintModelInfo() const;
    
    // Configuration management
    void SetDeploymentConfig(const DeploymentConfig& config);
    DeploymentConfig GetDeploymentConfig() const { return deploymentConfig; }
    
    // Model validation and testing
    bool ValidateModel();
    bool RunModelSelfTest();
    InferenceOptimizer::BenchmarkResults BenchmarkCurrentModel(int iterations = 100);
    
    // Error handling and fallbacks
    void SetFallbackModel(const std::string& modelPath, const std::string& configPath);
    bool HasFallbackModel() const { return fallbackModel != nullptr; }
    void EnableFallbackMode(bool enable) { useFallbackMode = enable; }
    
private:
    // Model storage
    std::unique_ptr<HierarchicalRTSModel> model;
    std::unique_ptr<HierarchicalRTSModel> fallbackModel;
    std::unordered_map<std::string, std::unique_ptr<HierarchicalRTSModel>> modelVariants;
    
    std::string currentModelName;
    std::string currentModelPath;
    std::string currentConfigPath;
    
    // Configuration
    DeploymentConfig deploymentConfig;
    bool useFallbackMode;
    
    // Async inference management
    std::atomic<bool> asyncInferenceEnabled;
    std::atomic<bool> shouldStopAsyncThread;
    std::unique_ptr<std::thread> asyncInferenceThread;
    
    std::queue<InferenceRequest> requestQueue;
    std::unordered_map<int, InferenceResult> resultMap;
    std::mutex requestMutex;
    std::mutex resultMutex;
    std::condition_variable requestCondition;
    
    std::atomic<int> nextRequestId;
    
    // Performance tracking
    mutable std::mutex statsMutex;
    PerformanceStats stats;
    std::vector<double> recentInferenceTimes;
    
    // Private methods
    void InitializeAsyncInference();
    void ShutdownAsyncInference();
    void AsyncInferenceThreadLoop();
    
    void UpdatePerformanceStats(double inferenceTime, bool success);
    void CleanupOldResults();
    
    MLOutput RunInferenceInternal(const SpatialTensor& input, bool isAsync = false);
    MLOutput RunFallbackInference(const SpatialTensor& input);
    
    bool ValidateInput(const SpatialTensor& input) const;
    void LogInferenceError(const std::string& error, bool useFallback = false);
    
    // Model loading helpers
    bool LoadModelInternal(const std::string& modelPath, const std::string& configPath);
    void OptimizeModelForDeployment();
};

/**
 * Model registry for managing multiple model configurations
 */
class ModelRegistry {
public:
    struct ModelInfo {
        std::string name;
        std::string description;
        std::string modelPath;
        std::string configPath;
        ModelConfig config;
        bool isAvailable;
        size_t sizeBytes;
    };
    
    static ModelRegistry& Instance();
    
    // Registry management
    void RegisterModel(const ModelInfo& info);
    void UnregisterModel(const std::string& name);
    bool IsModelRegistered(const std::string& name) const;
    
    // Model discovery
    std::vector<ModelInfo> GetAvailableModels() const;
    ModelInfo GetModelInfo(const std::string& name) const;
    std::vector<std::string> GetModelNames() const;
    
    // Model recommendations
    ModelInfo GetRecommendedModel(const DeploymentConfig& deployment) const;
    std::vector<ModelInfo> GetCompatibleModels(const DeploymentConfig& deployment) const;
    
    // Registry persistence
    void SaveRegistry(const std::string& registryPath) const;
    void LoadRegistry(const std::string& registryPath);
    
    // Auto-discovery
    void ScanDirectory(const std::string& modelDir);
    void RefreshRegistry();
    
private:
    std::unordered_map<std::string, ModelInfo> registry;
    mutable std::mutex registryMutex;
    
    ModelRegistry() = default;
    
    bool ValidateModelFiles(const ModelInfo& info) const;
    size_t GetFileSize(const std::string& filepath) const;
};

/**
 * Automatic model selection based on hardware and performance requirements
 */
class AutoModelSelector {
public:
    struct HardwareCapabilities {
        bool hasCUDA;
        size_t totalMemoryMB;
        size_t availableMemoryMB;
        int cudaComputeCapability;
        int numCPUCores;
        std::string cpuArchitecture;
    };
    
    struct PerformanceRequirements {
        float maxInferenceTimeMs;
        float targetThroughput;     // inferences per second
        size_t maxMemoryUsageMB;
        bool requireRealTime;
        float accuracyThreshold;    // minimum required accuracy
    };
    
    static HardwareCapabilities DetectHardware();
    static ModelRegistry::ModelInfo SelectOptimalModel(
        const HardwareCapabilities& hardware,
        const PerformanceRequirements& requirements);
    
    static DeploymentConfig CreateDeploymentConfig(
        const HardwareCapabilities& hardware,
        const PerformanceRequirements& requirements);
    
    // Performance prediction
    static double PredictInferenceTime(const ModelRegistry::ModelInfo& model,
                                      const HardwareCapabilities& hardware);
    static size_t PredictMemoryUsage(const ModelRegistry::ModelInfo& model);
    
    // Benchmark-based selection
    static ModelRegistry::ModelInfo SelectModelByBenchmark(
        const std::vector<ModelRegistry::ModelInfo>& candidates,
        const PerformanceRequirements& requirements);
};

} // namespace chanrts

#endif // _CHANRTS_MODELMANAGER_H