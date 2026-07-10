#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace py = pybind11;

class LGPA_Core {
private:
    int n;
    int m;
    std::vector<std::vector<int>> adj;
    std::vector<int> degree;
    std::vector<std::unordered_map<int, double>> sj; 
    std::vector<double> strength;
    double core_threshold;

public:
    LGPA_Core(int num_nodes, const std::vector<std::pair<int, int>>& edges) {
        n = num_nodes;
        m = edges.size();
        adj.resize(n);
        degree.resize(n, 0);
        sj.resize(n);
        strength.resize(n, 0.0);

        for (const auto& edge : edges) {
            int u = edge.first;
            int v = edge.second;
            adj[u].push_back(v);
            adj[v].push_back(u);
            degree[u]++;
            degree[v]++;
        }

        for (int i = 0; i < n; ++i) {
            std::sort(adj[i].begin(), adj[i].end());
        }

        std::vector<double> sj_values;
        sj_values.reserve(m);

        // STEP 1: Laplacian-Smoothed Jaccard
        for (const auto& edge : edges) {
            int u = edge.first;
            int v = edge.second;
            
            int common = 0;
            auto it_u = adj[u].begin();
            auto it_v = adj[v].begin();
            while (it_u != adj[u].end() && it_v != adj[v].end()) {
                if (*it_u < *it_v) ++it_u;
                else if (*it_v < *it_u) ++it_v;
                else {
                    common++;
                    ++it_u; ++it_v;
                }
            }
            
            int union_size = degree[u] + degree[v] - common;
            double sim = (common + 1.0) / (union_size + 1.0);
            sj[u][v] = sim;
            sj[v][u] = sim;
            sj_values.push_back(sim);
        }

        // STEP 2: Intrinsic Structural Strength
        for (int u = 0; u < n; ++u) {
            double s = 0.0;
            for (int v : adj[u]) {
                s += sj[u][v];
            }
            strength[u] = s;
        }

        // STEP 3: Statistical Regime Thresholds
        double mean_sj = 0.0, std_sj = 0.0, sci = 0.0;
        if (!sj_values.empty()) {
            double sum = std::accumulate(sj_values.begin(), sj_values.end(), 0.0);
            mean_sj = sum / sj_values.size();
            double sq_sum = std::inner_product(sj_values.begin(), sj_values.end(), sj_values.begin(), 0.0);
            // std::max guards against tiny negative variance from round-off.
            double var = std::max(0.0, sq_sum / sj_values.size() - mean_sj * mean_sj);
            std_sj = std::sqrt(var);
            sci = std_sj / mean_sj;  
        }

        double w_struct = 1.0 - std::exp(-sci);
        core_threshold = mean_sj + (w_struct * std_sj);
    }

    std::unordered_map<int, int> fit_predict(int max_iter) {
        if (n == 0) return {};

        // PHASE 1:  Coring
        std::vector<int> merged_to(n);
        std::iota(merged_to.begin(), merged_to.end(), 0);

        std::vector<int> nodes_desc(n);
        std::iota(nodes_desc.begin(), nodes_desc.end(), 0);
        std::sort(nodes_desc.begin(), nodes_desc.end(), [&](int a, int b) {
            if (strength[a] != strength[b]) return strength[a] > strength[b];
            return a > b; 
        });

        for (int u : nodes_desc) {
            if (adj[u].empty()) continue;

            int best_nbr = adj[u][0];
            double max_sj = sj[u][best_nbr];
            for (int v : adj[u]) {
                if (sj[u][v] > max_sj) {
                    max_sj = sj[u][v];
                    best_nbr = v;
                }
            }

            if (max_sj > core_threshold) {
                if (strength[best_nbr] > strength[u] || (strength[best_nbr] == strength[u] && best_nbr > u)) {
                    merged_to[u] = best_nbr;
                }
            }
        }

        // Path Compression
        for (int u = 0; u < n; ++u) {
            int curr = u;
            while (curr != merged_to[curr]) {
                curr = merged_to[curr];
            }
            merged_to[u] = curr;
        }

        // PHASE 2: Log-Gravity Propagation
        std::vector<int> labels = merged_to;
        std::vector<int> nodes_asc = nodes_desc;
        std::reverse(nodes_asc.begin(), nodes_asc.end());
        
        const double EULER_E = std::exp(1.0);

        for (int iter = 0; iter < max_iter; ++iter) {
            bool changed = false;
            for (int u : nodes_asc) {
                if (adj[u].empty()) continue;

                std::unordered_map<int, double> scores;
                for (int v : adj[u]) {
                    double weight = sj[u][v] * std::log(strength[v] + EULER_E);
                    scores[labels[v]] += weight;
                }

                if (scores.empty()) continue;

                int best_label = -1;
                double max_score = -1.0;
                for (const auto& pair : scores) {
                    if (pair.second > max_score) {
                        max_score = pair.second;
                        best_label = pair.first;
                    }
                }

                if (labels[u] != best_label) {
                    labels[u] = best_label;
                    changed = true;
                }
            }
            if (!changed) break;
        }

        std::unordered_map<int, int> final_labels;
        std::unordered_map<int, int> label_mapping;
        int current_id = 0;

        for (int u = 0; u < n; ++u) {
            int l = labels[u];
            if (label_mapping.find(l) == label_mapping.end()) {
                label_mapping[l] = current_id++;
            }
            final_labels[u] = label_mapping[l];
        }

        return final_labels;
    }
};

PYBIND11_MODULE(_lgpa_core, m) {
    py::class_<LGPA_Core>(m, "LGPA_Core")
        .def(py::init<int, const std::vector<std::pair<int, int>>&>())
        .def("fit_predict", &LGPA_Core::fit_predict, py::arg("max_iter") = 50, py::call_guard<py::gil_scoped_release>()); 
}