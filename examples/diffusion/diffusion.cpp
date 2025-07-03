// Diffusion generation example for masked language models like DiffuCoder
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-s steps] [-t temperature] [-a algorithm] [prompt]\n", argv[0]);
    printf("\n");
    printf("    -m model.gguf   : path to model file\n");
    printf("    -n n_predict    : number of tokens to generate (default: 32)\n");
    printf("    -s steps        : diffusion timesteps (default: 64)\n");
    printf("    -t temperature  : sampling temperature (default: 0.0)\n");
    printf("    -a algorithm    : remasking algorithm: origin, maskgit_plus, topk_margin, entropy (default: maskgit_plus)\n");
    printf("\n");
}

// Token confidence calculation strategies
enum class DiffusionAlgorithm {
    ORIGIN,        // Random selection
    MASKGIT_PLUS,  // Top-1 confidence
    TOPK_MARGIN,   // Top-1 minus Top-2
    ENTROPY        // Negative entropy
};

// Calculate token confidence based on logits
static std::vector<std::pair<int, float>> calculate_confidence(
    const float * logits,
    int n_vocab,
    int n_tokens,
    DiffusionAlgorithm alg,
    float temperature) {
    
    std::vector<std::pair<int, float>> confidence_scores;
    
    for (int i = 0; i < n_tokens; i++) {
        const float * token_logits = logits + i * n_vocab;
        
        // Apply temperature
        std::vector<float> scaled_logits(n_vocab);
        if (temperature > 0) {
            for (int j = 0; j < n_vocab; j++) {
                scaled_logits[j] = token_logits[j] / temperature;
            }
        } else {
            std::copy(token_logits, token_logits + n_vocab, scaled_logits.begin());
        }
        
        // Calculate softmax probabilities
        float max_logit = *std::max_element(scaled_logits.begin(), scaled_logits.end());
        std::vector<float> probs(n_vocab);
        float sum_exp = 0.0f;
        
        for (int j = 0; j < n_vocab; j++) {
            probs[j] = expf(scaled_logits[j] - max_logit);
            sum_exp += probs[j];
        }
        
        for (int j = 0; j < n_vocab; j++) {
            probs[j] /= sum_exp;
        }
        
        float confidence = 0.0f;
        
        switch (alg) {
            case DiffusionAlgorithm::ORIGIN:
                // Random confidence
                confidence = static_cast<float>(rand()) / RAND_MAX;
                break;
                
            case DiffusionAlgorithm::MASKGIT_PLUS:
                // Top-1 probability
                confidence = *std::max_element(probs.begin(), probs.end());
                break;
                
            case DiffusionAlgorithm::TOPK_MARGIN:
                // Top-1 minus Top-2
                {
                    std::vector<float> sorted_probs = probs;
                    std::sort(sorted_probs.begin(), sorted_probs.end(), std::greater<float>());
                    confidence = sorted_probs[0] - sorted_probs[1];
                }
                break;
                
            case DiffusionAlgorithm::ENTROPY:
                // Negative entropy
                confidence = 0.0f;
                for (float p : probs) {
                    if (p > 1e-10f) {
                        confidence += p * logf(p);
                    }
                }
                break;
        }
        
        confidence_scores.push_back({i, confidence});
    }
    
    return confidence_scores;
}

int main(int argc, char ** argv) {
    // Default parameters
    std::string model_path;
    std::string prompt;
    int n_predict = 32;
    int steps = 64;
    float temperature = 0.0f;
    DiffusionAlgorithm algorithm = DiffusionAlgorithm::MASKGIT_PLUS;
    int ngl = 99;
    
    // Parse command line arguments
    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    n_predict = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-s") == 0) {
                if (i + 1 < argc) {
                    steps = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-t") == 0) {
                if (i + 1 < argc) {
                    temperature = std::stof(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-a") == 0) {
                if (i + 1 < argc) {
                    std::string alg_str = argv[++i];
                    if (alg_str == "origin") {
                        algorithm = DiffusionAlgorithm::ORIGIN;
                    } else if (alg_str == "maskgit_plus") {
                        algorithm = DiffusionAlgorithm::MASKGIT_PLUS;
                    } else if (alg_str == "topk_margin") {
                        algorithm = DiffusionAlgorithm::TOPK_MARGIN;
                    } else if (alg_str == "entropy") {
                        algorithm = DiffusionAlgorithm::ENTROPY;
                    } else {
                        fprintf(stderr, "Unknown algorithm: %s\n", alg_str.c_str());
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    ngl = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // Prompt starts here
                break;
            }
        }
        
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }
    
    // Load dynamic backends
    ggml_backend_load_all();
    
    // Initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }
    
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    // Get special token IDs
    llama_token mask_token_id = llama_vocab_mask(vocab);
    if (mask_token_id == LLAMA_TOKEN_NULL) {
        fprintf(stderr, "%s: error: model does not have a mask token\n", __func__);
        return 1;
    }
    
    // Tokenize the prompt if provided
    std::vector<llama_token> tokens;
    if (!prompt.empty()) {
        int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
        tokens.resize(n_prompt);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true) < 0) {
            fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
            return 1;
        }
    }
    
    // Pad with mask tokens to reach target length
    int total_length = tokens.size() + n_predict;
    tokens.resize(total_length, mask_token_id);
    
    // Initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = total_length;
    ctx_params.n_batch = total_length;
    ctx_params.no_perf = false;
    
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }
    
    // Set non-causal attention for masked language modeling
    llama_set_causal_attn(ctx, false);
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Create timestep schedule
    std::vector<float> timesteps(steps + 1);
    float eps = 1e-3f;
    for (int i = 0; i <= steps; i++) {
        timesteps[i] = 1.0f - i * (1.0f - eps) / steps;
    }
    
    // Print initial state
    printf("Initial tokens (with masks):\n");
    for (llama_token id : tokens) {
        if (id == mask_token_id) {
            printf("[MASK]");
        } else {
            char buf[128];
            int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
            if (n > 0) {
                printf("%.*s", n, buf);
            }
        }
    }
    printf("\n\n");
    
    // Diffusion generation loop
    printf("Starting diffusion generation with %d steps...\n", steps);
    
    for (int step = 0; step < steps; step++) {
        float t = timesteps[step];
        float s = timesteps[step + 1];
        
        // Count mask tokens
        std::vector<int> mask_indices;
        for (int i = 0; i < total_length; i++) {
            if (tokens[i] == mask_token_id) {
                mask_indices.push_back(i);
            }
        }
        
        if (mask_indices.empty()) {
            printf("No more mask tokens to denoise.\n");
            break;
        }
        
        // Prepare batch
        llama_batch batch = llama_batch_get_one(tokens.data(), total_length);
        
        // Decode
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s: failed to decode\n", __func__);
            return 1;
        }
        
        // Get logits
        float * logits = llama_get_logits(ctx);
        int n_vocab = llama_model_n_vocab(model);
        
        // Calculate confidence scores for masked positions
        std::vector<std::pair<int, float>> mask_confidences;
        
        for (int idx : mask_indices) {
            const float * token_logits = logits + idx * n_vocab;
            
            // Apply temperature and calculate probabilities
            std::vector<float> probs(n_vocab);
            
            if (temperature > 0) {
                // Apply temperature scaling
                float max_logit = *std::max_element(token_logits, token_logits + n_vocab);
                float sum_exp = 0.0f;
                
                for (int j = 0; j < n_vocab; j++) {
                    probs[j] = expf((token_logits[j] - max_logit) / temperature);
                    sum_exp += probs[j];
                }
                
                for (int j = 0; j < n_vocab; j++) {
                    probs[j] /= sum_exp;
                }
            } else {
                // Greedy: take argmax
                int max_idx = std::distance(token_logits, std::max_element(token_logits, token_logits + n_vocab));
                std::fill(probs.begin(), probs.end(), 0.0f);
                probs[max_idx] = 1.0f;
            }
            
            // Sample token
            llama_token sampled_token;
            if (temperature > 0) {
                std::discrete_distribution<> dist(probs.begin(), probs.end());
                sampled_token = dist(gen);
            } else {
                sampled_token = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
            }
            
            // Calculate confidence based on algorithm
            float confidence = 0.0f;
            
            switch (algorithm) {
                case DiffusionAlgorithm::ORIGIN:
                    confidence = static_cast<float>(rand()) / RAND_MAX;
                    break;
                    
                case DiffusionAlgorithm::MASKGIT_PLUS:
                    confidence = probs[sampled_token];
                    break;
                    
                case DiffusionAlgorithm::TOPK_MARGIN:
                    {
                        std::vector<float> sorted_probs = probs;
                        std::sort(sorted_probs.begin(), sorted_probs.end(), std::greater<float>());
                        confidence = sorted_probs[0] - sorted_probs[1];
                    }
                    break;
                    
                case DiffusionAlgorithm::ENTROPY:
                    confidence = 0.0f;
                    for (float p : probs) {
                        if (p > 1e-10f) {
                            confidence += p * logf(p);
                        }
                    }
                    break;
            }
            
            mask_confidences.push_back({idx, confidence});
            
            // Store the sampled token temporarily
            mask_confidences.back().first = (mask_confidences.back().first << 16) | sampled_token;
        }
        
        // Sort by confidence (descending)
        std::sort(mask_confidences.begin(), mask_confidences.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Calculate number of tokens to unmask
        int num_unmask = static_cast<int>(mask_indices.size() * (1.0f - s / t));
        if (step == steps - 1) {
            num_unmask = mask_indices.size(); // Unmask all remaining tokens
        }
        
        // Unmask the most confident tokens
        for (int i = 0; i < num_unmask && i < mask_confidences.size(); i++) {
            int idx = mask_confidences[i].first >> 16;
            llama_token token = mask_confidences[i].first & 0xFFFF;
            tokens[idx] = token;
        }
        
        // Print progress
        if ((step + 1) % 10 == 0 || step == steps - 1) {
            printf("\nStep %d/%d (%.1f%% unmasked):\n", step + 1, steps, 
                   100.0f * (total_length - mask_indices.size() + num_unmask) / total_length);
            
            for (llama_token id : tokens) {
                if (id == mask_token_id) {
                    printf("[MASK]");
                } else {
                    char buf[128];
                    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
                    if (n > 0) {
                        printf("%.*s", n, buf);
                    }
                }
            }
            printf("\n");
        }
        
        // Clear KV cache for next iteration
        llama_kv_cache_clear(ctx);
    }
    
    // Print final result
    printf("\nFinal result:\n");
    for (llama_token id : tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            printf("%.*s", n, buf);
        }
    }
    printf("\n");
    
    // Cleanup
    llama_free(ctx);
    llama_model_free(model);
    
    return 0;
}