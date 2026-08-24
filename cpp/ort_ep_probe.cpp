#include <onnxruntime_cxx_api.h>
#include <spacemit_ort_env.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string model;
  std::string ep = "spacemit";
  std::string affinity;
  int intra_threads = 1;
  int inter_threads = 1;
  int warmup = 0;
  int iters = 1;
  int severity = 2;
  bool run = false;
};

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " --model MODEL [options]\n"
            << "  --ep spacemit|cpu       Execution provider (default: spacemit)\n"
            << "  --affinity IDS          SpaceMIT EP affinity, e.g. 8 or 8;9\n"
            << "  --intra-threads N       SpaceMIT EP intra thread count\n"
            << "  --inter-threads N       SpaceMIT EP inter thread count\n"
            << "  --warmup N              Warmup Run() calls (default: 0)\n"
            << "  --iters N               Measured Run() calls (default: 1)\n"
            << "  --run                   Try a zero-input smoke Run()\n"
            << "  --severity N            ORT log severity 0..4 (default: 2)\n";
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (key == "--model") a.model = need("--model");
    else if (key == "--ep") a.ep = need("--ep");
    else if (key == "--affinity") a.affinity = need("--affinity");
    else if (key == "--intra-threads") a.intra_threads = std::stoi(need("--intra-threads"));
    else if (key == "--inter-threads") a.inter_threads = std::stoi(need("--inter-threads"));
    else if (key == "--warmup") a.warmup = std::stoi(need("--warmup"));
    else if (key == "--iters") a.iters = std::stoi(need("--iters"));
    else if (key == "--severity") a.severity = std::stoi(need("--severity"));
    else if (key == "--run") a.run = true;
    else if (key == "--help" || key == "-h") { usage(argv[0]); std::exit(0); }
    else throw std::runtime_error("unknown option: " + key);
  }
  if (a.model.empty()) throw std::runtime_error("--model is required");
  if (a.ep != "cpu" && a.ep != "spacemit") throw std::runtime_error("--ep must be cpu or spacemit");
  if (a.warmup < 0 || a.iters <= 0) throw std::runtime_error("warmup >= 0 and iters > 0 required");
  return a;
}

template <typename TensorInfo>
std::string shape_to_string(const TensorInfo& info) {
  std::ostringstream os;
  os << "[";
  const auto shape = info.GetShape();
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) os << ",";
    os << shape[i];
  }
  os << "]";
  return os.str();
}

void print_value_info(const char* side, const std::vector<Ort::ValueInfo>& infos) {
  std::cout << side << "_count=" << infos.size() << "\n";
  for (size_t i = 0; i < infos.size(); ++i) {
    const auto& info = infos[i];
    std::cout << side << "[" << i << "] name=" << info.GetName()
              << " type=" << info.TypeInfo().GetONNXType();
    if (info.TypeInfo().GetONNXType() == ONNX_TYPE_TENSOR) {
      const auto tensor = info.TypeInfo().GetTensorTypeAndShapeInfo();
      std::cout << " elem=" << tensor.GetElementType() << " shape=" << shape_to_string(tensor);
    }
    std::cout << "\n";
  }
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string item;
  std::stringstream ss(s);
  while (std::getline(ss, item, sep)) if (!item.empty()) out.push_back(item);
  return out;
}

void print_assignments(const Ort::Session& session) {
  const auto assignments = session.GetEpGraphAssignmentInfo();
  std::cout << "ep_assignment_count=" << assignments.size() << "\n";
  for (size_t i = 0; i < assignments.size(); ++i) {
    const auto& subgraph = assignments[i];
    std::cout << "ep_assignment[" << i << "] ep=" << subgraph.GetEpName()
              << " nodes=" << subgraph.GetNodes().size() << "\n";
  }
}

template <typename TensorInfo>
std::vector<int64_t> concrete_shape(const TensorInfo& info) {
  auto shape = info.GetShape();
  for (auto& dim : shape) {
    if (dim <= 0) dim = 1;
  }
  // Avoid accidental huge allocations for unknown sequence dimensions.
  size_t elements = 1;
  for (auto dim : shape) {
    if (dim <= 0 || static_cast<uint64_t>(dim) > 4096 || elements > 16 * 1024 * 1024 / static_cast<size_t>(dim)) {
      return {};
    }
    elements *= static_cast<size_t>(dim);
  }
  return shape;
}

// Keeps the host backing store alive for the full Session::Run() lifetime.
// Ort::Value::CreateTensor() does not own the passed buffer.
struct OwnedInput {
  std::vector<float> f32;
  std::vector<int64_t> i64;
  std::vector<int32_t> i32;
  Ort::Value value{nullptr};
};

std::unique_ptr<OwnedInput> make_zero_input(const Ort::MemoryInfo& memory_info, const Ort::ValueInfo& info) {
  const auto type = info.TypeInfo().GetONNXType();
  if (type != ONNX_TYPE_TENSOR) throw std::runtime_error("non-tensor input cannot be auto-filled: " + std::string(info.GetName()));
  const auto tensor = info.TypeInfo().GetTensorTypeAndShapeInfo();
  const auto shape = concrete_shape(tensor);
  if (shape.empty()) throw std::runtime_error("dynamic/large input requires explicit input: " + std::string(info.GetName()));
  const auto elem = tensor.GetElementType();
  size_t count = 1;
  for (auto dim : shape) count *= static_cast<size_t>(dim);

  auto owned = std::make_unique<OwnedInput>();
  if (elem == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    owned->f32.assign(count, 0.0f);
    owned->value = Ort::Value::CreateTensor<float>(memory_info, owned->f32.data(), owned->f32.size(), shape.data(), shape.size());
  } else if (elem == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    owned->i64.assign(count, 0);
    owned->value = Ort::Value::CreateTensor<int64_t>(memory_info, owned->i64.data(), owned->i64.size(), shape.data(), shape.size());
  } else if (elem == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    owned->i32.assign(count, 0);
    owned->value = Ort::Value::CreateTensor<int32_t>(memory_info, owned->i32.data(), owned->i32.size(), shape.data(), shape.size());
  } else {
    throw std::runtime_error("unsupported auto-fill input element type for " + std::string(info.GetName()));
  }
  return owned;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse(argc, argv);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "model=" << args.model << " ep=" << args.ep << "\n";
    std::cout << "available_providers=";
    for (const auto& provider : Ort::GetAvailableProviders()) std::cout << provider << ",";
    std::cout << "\n";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "moss-tts-ort-probe");
    Ort::SessionOptions options;
    options.SetLogSeverityLevel(args.severity);
    options.SetIntraOpNumThreads(args.intra_threads);
    options.SetInterOpNumThreads(args.inter_threads);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.AddConfigEntry("session.record_ep_graph_assignment_info", "1");

    if (args.ep == "spacemit") {
      std::unordered_map<std::string, std::string> ep_options;
      if (!args.affinity.empty()) ep_options["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = args.affinity;
      ep_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(args.intra_threads);
      ep_options["SPACEMIT_EP_INTER_THREAD_NUM"] = std::to_string(args.inter_threads);
      auto status = Ort::SessionOptionsSpaceMITEnvInit(options, ep_options);
      if (!status.IsOK()) throw std::runtime_error("SessionOptionsSpaceMITEnvInit: " + std::string(status.GetErrorMessage()));
      std::cout << "spacemit_ep_init=ok\n";
    } else {
      std::cout << "cpu_ep_init=default\n";
    }

    const auto create_start = std::chrono::steady_clock::now();
    Ort::Session session(env, args.model.c_str(), options);
    const auto create_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_start).count();
    std::cout << "session_create_ms=" << create_ms << "\n";
    print_value_info("input", session.GetInputs());
    print_value_info("output", session.GetOutputs());
    print_assignments(session);

    if (!args.run) return 0;

    const std::vector<Ort::ValueInfo> input_infos = session.GetInputs();
    const std::vector<Ort::ValueInfo> output_infos = session.GetOutputs();
    std::vector<const char*> input_names;
    std::vector<std::string> input_name_storage;
    std::vector<std::unique_ptr<OwnedInput>> owned_inputs;
    std::vector<Ort::Value> input_values;
    input_names.reserve(input_infos.size());
    input_name_storage.reserve(input_infos.size());
    owned_inputs.reserve(input_infos.size());
    input_values.reserve(input_infos.size());
    const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (const auto& info : input_infos) {
      input_name_storage.push_back(info.GetName());
      owned_inputs.push_back(make_zero_input(memory_info, info));
      input_names.push_back(input_name_storage.back().c_str());
      input_values.push_back(std::move(owned_inputs.back()->value));
    }
    std::vector<const char*> output_names;
    std::vector<std::string> output_name_storage;
    output_names.reserve(output_infos.size());
    output_name_storage.reserve(output_infos.size());
    for (const auto& info : output_infos) {
      output_name_storage.push_back(info.GetName());
      output_names.push_back(output_name_storage.back().c_str());
    }

    for (int i = 0; i < args.warmup; ++i) {
      session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(), input_values.size(), output_names.data(), output_names.size());
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < args.iters; ++i) {
      auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(), input_values.size(), output_names.data(), output_names.size());
      (void)outputs;
    }
    const double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "run_iters=" << args.iters << " total_ms=" << elapsed_ms << " avg_ms=" << (elapsed_ms / args.iters) << "\n";
    return 0;
  } catch (const Ort::Exception& e) {
    std::cerr << "ORT_EXCEPTION: " << e.what() << "\n";
    return 20;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 21;
  }
}
