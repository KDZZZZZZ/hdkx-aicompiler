#pragma once

#include "base/capi.hpp"
#include "node.hpp"
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <regex>
#include <algorithm>
#include <cctype>

namespace kxcomp{

// Helper functions for parsing arguments
static inline void trim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

static std::vector<int32_t> parse_int_vector(const std::string& s) {
    std::vector<int32_t> result;
    std::string list_str = s;
    list_str.erase(std::remove(list_str.begin(), list_str.end(), '['), list_str.end());
    list_str.erase(std::remove(list_str.begin(), list_str.end(), ']'), list_str.end());
    
    std::stringstream ss(list_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        trim(item);
        if(!item.empty())
            result.push_back(std::stoi(item));
    }
    return result;
}

static void parse_args(const std::string& args_str, 
              std::vector<std::string>& pos_args, 
              std::unordered_map<std::string, std::string>& kw_args) {
    std::string current_arg;
    int bracket_level = 0;
    for (char c : args_str + ",") {
        if (c == ',' && bracket_level == 0) {
            trim(current_arg);
            if (current_arg.empty()) continue;

            size_t eq_pos = current_arg.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = current_arg.substr(0, eq_pos);
                std::string value = current_arg.substr(eq_pos + 1);
                trim(key);
                trim(value);
                kw_args[key] = value;
            } else {
                pos_args.push_back(current_arg);
            }
            current_arg = "";
        } else {
            if (c == '[' || c == '(') bracket_level++;
            if (c == ']' || c == ')') bracket_level--;
            current_arg += c;
        }
    }
}

class GraphPtr:public objectPtr<Graph>{//TODO:支持子图replace
    SIMPLE_DECLARE_TYPE(GraphPtr,objectPtr<Graph>)
public:
    GraphPtr() = default;
};

template<> GraphPtr make_object<graph>(const std::string& data) {
    graph* ptr = new graph(data);
    const int32_t type_index = graph::RuntimeTypeIndex();
    ptr->SetTypeIndex(type_index);
    ptr->SetDeleter([](void* obj) { delete static_cast<graph*>(obj); });
    return Graphptr(ptr);
}

SIMPLE_REGISTER_TYPE(Graph)
class Graph:public object{
    SIMPLE_DECLARE_TYPE(Graph,object)
public:
    std::vector<NodePtr> graph_nodes;
    void add_node(NodePtr node_to_add){
        // 1. 将待增加节点加入 graph_nodes
        // 检查节点是否已存在，避免重复添加相同的底层Node对象
        auto it_graph = std::find_if(graph_nodes.begin(), graph_nodes.end(), 
                                       [&](const NodePtr& n_ptr){ return n_ptr.get() == node_to_add.get(); });
        if (it_graph == graph_nodes.end()) {
            graph_nodes.push_back(node_to_add);
        } else {
            // 如果节点已存在，可能需要决定如何处理，这里选择直接返回
            return; 
        }

        // 2. 完善新节点的进出边结构：更新其前驱和后继节点的连接
        
        // 更新前驱节点的 next_nodes，使其包含 node_to_add
        for(auto& prev_node_ref : node_to_add.prev_nodes){
            auto it_actual_prev = std::find_if(graph_nodes.begin(), graph_nodes.end(), 
                                               [&](const NodePtr& n_ptr){ return n_ptr.get() == prev_node_ref.get(); });
            if (it_actual_prev != graph_nodes.end()) {
                NodePtr& actual_prev_node = *it_actual_prev;
                // 确保不重复添加
                auto it_exists = std::find_if(actual_prev_node.next_nodes.begin(), actual_prev_node.next_nodes.end(),
                                               [&](const NodePtr& n_ptr){ return n_ptr.get() == node_to_add.get(); });
                if (it_exists == actual_prev_node.next_nodes.end()) {
                    actual_prev_node.next_nodes.push_back(node_to_add);
                }
            } else {
                // 如果前驱节点不在 graph_nodes 中，这可能表示图结构不一致。
                // 此时，无法更新其 next_nodes。
            }
        }

        // 更新后继节点的 prev_nodes，使其包含 node_to_add
        for(auto& next_node_ref : node_to_add.next_nodes){
            auto it_actual_next = std::find_if(graph_nodes.begin(), graph_nodes.end(), 
                                               [&](const NodePtr& n_ptr){ return n_ptr.get() == next_node_ref.get(); });
            if (it_actual_next != graph_nodes.end()) {
                NodePtr& actual_next_node = *it_actual_next;
                // 确保不重复添加
                auto it_exists = std::find_if(actual_next_node.prev_nodes.begin(), actual_next_node.prev_nodes.end(),
                                               [&](const NodePtr& n_ptr){ return n_ptr.get() == node_to_add.get(); });
                if (it_exists == actual_next_node.prev_nodes.end()) {
                    actual_next_node.prev_nodes.push_back(node_to_add);
                }
            } else {
                // 如果后继节点不在 graph_nodes 中，这可能表示图结构不一致。
                // 此时，无法更新其 prev_nodes。
            }
        }
    }
    void remove_node(NodePtr node_to_remove){
        // 1. 从 graph_nodes 向量中移除 node_to_remove
        auto it_graph = std::remove_if(graph_nodes.begin(), graph_nodes.end(), 
                                       [&](const NodePtr& n_ptr){ return n_ptr.get() == node_to_remove.get(); });
        graph_nodes.erase(it_graph, graph_nodes.end());

        // 获取要移除节点的前驱节点和后继节点。
        std::vector<NodePtr> prev_nodes_of_removed = node_to_remove.prev_nodes;
        std::vector<NodePtr> next_nodes_of_removed = node_to_remove.next_nodes;

        // 2. 删除所有与 node_to_remove 相关的边（不连接前驱和后继）

        // 遍历被移除节点的所有前驱节点
        for(auto& prev_node_ref : prev_nodes_of_removed){
            // 找到 graph_nodes 中与 prev_node_ref 对应的"实际" NodePtr 实例
            auto it_actual_prev = std::find_if(graph_nodes.begin(), graph_nodes.end(), 
                                               [&](const NodePtr& n_ptr){ return n_ptr.get() == prev_node_ref.get(); });

            if (it_actual_prev != graph_nodes.end()) {
                NodePtr& actual_prev_node = *it_actual_prev;

                // 从其后继节点列表中移除 node_to_remove
                auto& p_next_nodes = actual_prev_node.next_nodes;
                auto it_rm_from_p = std::remove_if(p_next_nodes.begin(), p_next_nodes.end(), 
                                                   [&](const NodePtr& n_ptr){ return n_ptr.get() == node_to_remove.get(); });
                p_next_nodes.erase(it_rm_from_p, p_next_nodes.end());
            }
        }

        // 遍历被移除节点的所有后继节点
        for(auto& next_node_ref : next_nodes_of_removed){
            // 找到 graph_nodes 中与 next_node_ref 对应的"实际" NodePtr 实例
            auto it_actual_next = std::find_if(graph_nodes.begin(), graph_nodes.end(), 
                                               [&](const NodePtr& n_ptr){ return n_ptr.get() == next_node_ref.get(); });

            if (it_actual_next != graph_nodes.end()) {
                NodePtr& actual_next_node = *it_actual_next;

                // 从其前驱节点列表中移除 node_to_remove
                auto& n_prev_nodes = actual_next_node.prev_nodes;
                auto it_rm_from_n = std::remove_if(n_prev_nodes.begin(), n_prev_nodes.end(), 
                                                   [&](const NodePtr& n_ptr){ return n_ptr.get() == node_to_remove.get(); });
                n_prev_nodes.erase(it_rm_from_n, n_prev_nodes.end());
            }
        }
    }
    void replace_node(NodePtr node_to_replace,NodePtr new_node){//TODO:增加结构哈希验证
        // 为了方便操作，将 objectPtr<Node> 转换为 NodePtr

        // 1. 获取被替换节点的进出边信息
        std::vector<NodePtr> prev_nodes_of_old = node_to_replace.prev_nodes;
        std::vector<NodePtr> next_nodes_of_old = node_to_replace.next_nodes;

        // 2. 从 graph_nodes 中移除旧节点
        auto it_remove_old = std::remove_if(graph_nodes.begin(), graph_nodes.end(), 
                                            [&](const NodePtr& n_ptr){ return n_ptr.get() == node_to_replace.get(); });
        graph_nodes.erase(it_remove_old, graph_nodes.end());

        // 3. 将新节点添加到 graph_nodes
        // 确保新节点具有与旧节点相同的进出边信息，这样在添加到 graph_nodes 后，
        // 其内部的 next_nodes 和 prev_nodes 列表就能反映出继承的结构。
        new_node.prev_nodes = prev_nodes_of_old;
        new_node.next_nodes = next_nodes_of_old;
        graph_nodes.push_back(new_node); // 这里添加的是带有继承边信息的新节点

        // 4. 更新旧节点的前驱和后继，使其指向新节点（不改变进出边结构）

        // 更新旧节点的前驱，使其 next_nodes 指向新节点
        for(auto& prev_node_ref : prev_nodes_of_old){
            auto it_actual_prev = std::find_if(graph_nodes.begin(), graph_nodes.end(), 
                                               [&](const NodePtr& n_ptr){ return n_ptr.get() == prev_node_ref.get(); });
            if (it_actual_prev != graph_nodes.end()) {
                NodePtr& actual_prev_node = *it_actual_prev;
                // 从其 next_nodes 中移除旧节点
                auto& p_next_nodes = actual_prev_node.next_nodes;
                auto it_rm_old = std::remove_if(p_next_nodes.begin(), p_next_nodes.end(), 
                                                 [&](const NodePtr& n_ptr){ return n_ptr.get() == node_to_replace.get(); });
                p_next_nodes.erase(it_rm_old, p_next_nodes.end());
                // 将新节点添加到其 next_nodes 中
                p_next_nodes.push_back(new_node);
            }
        }

        // 更新旧节点的后继，使其 prev_nodes 指向新节点
        for(auto& next_node_ref : next_nodes_of_old){
            auto it_actual_next = std::find_if(graph_nodes.begin(), graph_nodes.end(), 
                                               [&](const NodePtr& n_ptr){ return n_ptr.get() == next_node_ref.get(); });
            if (it_actual_next != graph_nodes.end()) {
                NodePtr& actual_next_node = *it_actual_next;
                // 从其 prev_nodes 中移除旧节点
                auto& n_prev_nodes = actual_next_node.prev_nodes;
                auto it_rm_old = std::remove_if(n_prev_nodes.begin(), n_prev_nodes.end(), 
                                                 [&](const NodePtr& n_ptr){ return n_ptr.get() == node_to_replace.get(); });
                n_prev_nodes.erase(it_rm_old, n_prev_nodes.end());
                // 将新节点添加到其 prev_nodes 中
                n_prev_nodes.push_back(new_node);
            }
        }
    }
    void clear_graph_nodes(){
        this->graph_nodes.clear();
    }
    
    std::string graph_name;
    Graph() = default;

    Graph(const std::string& file_path) {//TODO：检查命名规则
        this->graph_name = file_path;
        std::string content = file_op::read_file(file_path);
        if (content.empty()) {
            return;
        }

        std::unordered_map<std::string, NodePtr> name_to_node;
        std::stringstream ss(content);
        std::string line;

        const std::regex line_regex(R"(\s*(\w+)\s*=\s*(\w+)\s*\((.*)\)\s*)");

        while (std::getline(ss, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }

            std::smatch match;
            if (std::regex_match(line, match, line_regex)) {
                std::string output_name = match[1];
                std::string op_type = match[2];
                std::string args_str = match[3];

                std::vector<std::string> pos_args;
                std::unordered_map<std::string, std::string> kw_args;
                parse_args(args_str, pos_args, kw_args);
                
                Node* new_node_raw = nullptr;

                // Node Factory
                if (op_type == "Input") {
                    new_node_raw = make_object<Input>(output_name, nullptr).release();
                } else if (op_type == "Output") {
                    new_node_raw = make_object<Output>(output_name, nullptr).release();
                } else if (op_type == "Gather" && !pos_args.empty()) {
                    new_node_raw = make_object<Gather>(output_name, pos_args[0], pos_args.size() > 1 ? pos_args[1] : "").release();
                } else if (op_type == "Unsqueeze" && !pos_args.empty() && kw_args.count("axes")) {
                    new_node_raw = make_object<Unsqueeze>(output_name, pos_args[0], parse_int_vector(kw_args["axes"])).release();
                } else if (op_type == "Slice" && !pos_args.empty()) {
                    new_node_raw = make_object<Slice>(output_name, pos_args[0], parse_int_vector(kw_args["starts"]), parse_int_vector(kw_args["ends"]), parse_int_vector(kw_args["axes"]), parse_int_vector(kw_args["steps"])).release();
                } else if (op_type == "Add" && pos_args.size() == 2) {
                    new_node_raw = make_object<Add>(output_name, pos_args[0], pos_args[1]).release();
                } else if (op_type == "ReduceMean" && !pos_args.empty()) {
                    new_node_raw = make_object<ReduceMean>(output_name, pos_args[0], parse_int_vector(kw_args["axes"]), kw_args["keep_dims"] == "true").release();
                } else if (op_type == "Div" && pos_args.size() == 2) {
                    new_node_raw = make_object<Div>(output_name, pos_args[0], pos_args[1]).release();
                } else if (op_type == "Pow" && !pos_args.empty()) {
                    new_node_raw = make_object<Pow>(output_name, pos_args[0], std::stoi(kw_args["power"])).release();
                } else if (op_type == "Sqrt" && !pos_args.empty()) {
                    new_node_raw = make_object<Sqrt>(output_name, pos_args[0]).release();
                } else if (op_type == "Mul" && pos_args.size() == 2) {
                    new_node_raw = make_object<Mul>(output_name, pos_args[0], pos_args[1]).release();
                } else if (op_type == "MatMul" && pos_args.size() == 2) {
                    new_node_raw = make_object<MatMul>(output_name, pos_args[0], pos_args[1]).release();
                } else if (op_type == "Concat" && !pos_args.empty()) {
                    new_node_raw = make_object<Concat>(output_name, pos_args, std::stoi(kw_args["axis"])).release();
                } else if (op_type == "Reshape" && !pos_args.empty()) {
                    new_node_raw = make_object<Reshape>(output_name, pos_args[0], parse_int_vector(kw_args["shape"]), kw_args["allow_zero"] == "true").release();
                } else if (op_type == "Transpose" && !pos_args.empty()) {
                    new_node_raw = make_object<Transpose>(output_name, pos_args[0], parse_int_vector(kw_args["perm"])).release();
                } else if (op_type == "Cast" && !pos_args.empty()) {
                    new_node_raw = make_object<Cast>(output_name, pos_args[0], kw_args["to"]).release();
                } else if (op_type == "Softmax" && !pos_args.empty()) {
                    new_node_raw = make_object<Softmax>(output_name, pos_args[0], std::stoi(kw_args["axis"])).release();
                } else if (op_type == "Split" && !pos_args.empty()) {
                    new_node_raw = make_object<Split>(output_name, pos_args[0], parse_int_vector(kw_args["split"]), std::stoi(kw_args["axis"])).release();
                } else if (op_type == "Constent") {
                    new_node_raw = make_object<Constent>(output_name, nullptr).release();
                } else if (op_type == "Shape") {
                    new_node_raw = make_object<Shape>(output_name, pos_args[0]).release();
                } else if (op_type == "Sub") {
                    new_node_raw = make_object<Sub>(output_name, pos_args[0], pos_args[1]).release();
                }

                if (new_node_raw) {
                    NodePtr new_node_ptr(new_node_raw);
                    
                    for (const auto& name : pos_args) {
                        if (name_to_node.count(name)) {
                            new_node_ptr.prev_nodes.push_back(name_to_node[name]);
                        }
                    }

                    add_node(new_node_ptr);
                    name_to_node[output_name] = new_node_ptr;
                }
            }
        }
    }

    void save_graph(const std::string& file_path) {
        std::string content = "";
        
        // For simplicity, we iterate through nodes as they are.
        // For better readability of the output file, a topological sort would be ideal
        // to ensure nodes are defined before they are used as inputs.
        // However, the current loader can handle out-of-order definitions.
        for (const auto& node_ptr : this->graph_nodes) {
            Node* node = node_ptr.get();
            if (!node) continue;

            content += node->name + " = " + node->op_type + "(";

            std::string args_str = "";
            for (size_t i = 0; i < node_ptr.prev_nodes.size(); ++i) {
                if(auto prev_node = node_ptr.prev_nodes[i].get()){
                    args_str += prev_node->name;
                    if (i < node_ptr.prev_nodes.size() - 1) {
                        args_str += ", ";
                    }
                }
            }
            content += args_str + ")\n";
        }
        
        file_op::write_file(file_path, content);
    }
    
};

}