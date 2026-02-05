#include "../include/relay/relay.h"
#include "../include/relay/op.h"
#include <vector>
#include <iostream>
#include <string>

using namespace kxc;

Expr build_graph() {
    // --- Initializers (Constants) ---
    Tensor const_0_tensor({30522, 768}, "float32");
    Constant const_0(const_0_tensor);
    Tensor const_1_tensor({512, 768}, "float32");
    Constant const_1(const_1_tensor);
    Tensor const_2_tensor({768}, "float32");
    Constant const_2(const_2_tensor);
    Tensor const_3_tensor({768}, "float32");
    Constant const_3(const_3_tensor);
    Tensor const_4_tensor({768}, "float32");
    Constant const_4(const_4_tensor);
    Tensor const_5_tensor({768}, "float32");
    Constant const_5(const_5_tensor);
    Tensor const_6_tensor({768}, "float32");
    Constant const_6(const_6_tensor);
    Tensor const_7_tensor({768}, "float32");
    Constant const_7(const_7_tensor);
    Tensor const_8_tensor({768}, "float32");
    Constant const_8(const_8_tensor);
    Tensor const_9_tensor({768}, "float32");
    Constant const_9(const_9_tensor);
    Tensor const_10_tensor({3072}, "float32");
    Constant const_10(const_10_tensor);
    Tensor const_11_tensor({768}, "float32");
    Constant const_11(const_11_tensor);
    Tensor const_12_tensor({768}, "float32");
    Constant const_12(const_12_tensor);
    Tensor const_13_tensor({768}, "float32");
    Constant const_13(const_13_tensor);
    Tensor const_14_tensor({768}, "float32");
    Constant const_14(const_14_tensor);
    Tensor const_15_tensor({768}, "float32");
    Constant const_15(const_15_tensor);
    Tensor const_16_tensor({768}, "float32");
    Constant const_16(const_16_tensor);
    Tensor const_17_tensor({768}, "float32");
    Constant const_17(const_17_tensor);
    Tensor const_18_tensor({768}, "float32");
    Constant const_18(const_18_tensor);
    Tensor const_19_tensor({768}, "float32");
    Constant const_19(const_19_tensor);
    Tensor const_20_tensor({3072}, "float32");
    Constant const_20(const_20_tensor);
    Tensor const_21_tensor({768}, "float32");
    Constant const_21(const_21_tensor);
    Tensor const_22_tensor({768}, "float32");
    Constant const_22(const_22_tensor);
    Tensor const_23_tensor({768}, "float32");
    Constant const_23(const_23_tensor);
    Tensor const_24_tensor({768}, "float32");
    Constant const_24(const_24_tensor);
    Tensor const_25_tensor({768}, "float32");
    Constant const_25(const_25_tensor);
    Tensor const_26_tensor({768}, "float32");
    Constant const_26(const_26_tensor);
    Tensor const_27_tensor({768}, "float32");
    Constant const_27(const_27_tensor);
    Tensor const_28_tensor({768}, "float32");
    Constant const_28(const_28_tensor);
    Tensor const_29_tensor({768}, "float32");
    Constant const_29(const_29_tensor);
    Tensor const_30_tensor({3072}, "float32");
    Constant const_30(const_30_tensor);
    Tensor const_31_tensor({768}, "float32");
    Constant const_31(const_31_tensor);
    Tensor const_32_tensor({768}, "float32");
    Constant const_32(const_32_tensor);
    Tensor const_33_tensor({768}, "float32");
    Constant const_33(const_33_tensor);
    Tensor const_34_tensor({768}, "float32");
    Constant const_34(const_34_tensor);
    Tensor const_35_tensor({768}, "float32");
    Constant const_35(const_35_tensor);
    Tensor const_36_tensor({768}, "float32");
    Constant const_36(const_36_tensor);
    Tensor const_37_tensor({768}, "float32");
    Constant const_37(const_37_tensor);
    Tensor const_38_tensor({768}, "float32");
    Constant const_38(const_38_tensor);
    Tensor const_39_tensor({768}, "float32");
    Constant const_39(const_39_tensor);
    Tensor const_40_tensor({3072}, "float32");
    Constant const_40(const_40_tensor);
    Tensor const_41_tensor({768}, "float32");
    Constant const_41(const_41_tensor);
    Tensor const_42_tensor({768}, "float32");
    Constant const_42(const_42_tensor);
    Tensor const_43_tensor({768}, "float32");
    Constant const_43(const_43_tensor);
    Tensor const_44_tensor({768}, "float32");
    Constant const_44(const_44_tensor);
    Tensor const_45_tensor({768}, "float32");
    Constant const_45(const_45_tensor);
    Tensor const_46_tensor({768}, "float32");
    Constant const_46(const_46_tensor);
    Tensor const_47_tensor({768}, "float32");
    Constant const_47(const_47_tensor);
    Tensor const_48_tensor({768}, "float32");
    Constant const_48(const_48_tensor);
    Tensor const_49_tensor({768}, "float32");
    Constant const_49(const_49_tensor);
    Tensor const_50_tensor({3072}, "float32");
    Constant const_50(const_50_tensor);
    Tensor const_51_tensor({768}, "float32");
    Constant const_51(const_51_tensor);
    Tensor const_52_tensor({768}, "float32");
    Constant const_52(const_52_tensor);
    Tensor const_53_tensor({768}, "float32");
    Constant const_53(const_53_tensor);
    Tensor const_54_tensor({768}, "float32");
    Constant const_54(const_54_tensor);
    Tensor const_55_tensor({768}, "float32");
    Constant const_55(const_55_tensor);
    Tensor const_56_tensor({768}, "float32");
    Constant const_56(const_56_tensor);
    Tensor const_57_tensor({768}, "float32");
    Constant const_57(const_57_tensor);
    Tensor const_58_tensor({768}, "float32");
    Constant const_58(const_58_tensor);
    Tensor const_59_tensor({768}, "float32");
    Constant const_59(const_59_tensor);
    Tensor const_60_tensor({3072}, "float32");
    Constant const_60(const_60_tensor);
    Tensor const_61_tensor({768}, "float32");
    Constant const_61(const_61_tensor);
    Tensor const_62_tensor({768}, "float32");
    Constant const_62(const_62_tensor);
    Tensor const_63_tensor({768}, "float32");
    Constant const_63(const_63_tensor);
    Tensor const_64_tensor({2}, "float32");
    Constant const_64(const_64_tensor);
    Tensor const_65_tensor({768, 768}, "float32");
    Constant const_65(const_65_tensor);
    Tensor const_66_tensor({768, 768}, "float32");
    Constant const_66(const_66_tensor);
    Tensor const_67_tensor({768, 768}, "float32");
    Constant const_67(const_67_tensor);
    Tensor const_68_tensor({768, 768}, "float32");
    Constant const_68(const_68_tensor);
    Tensor const_69_tensor({768, 3072}, "float32");
    Constant const_69(const_69_tensor);
    Tensor const_70_tensor({3072, 768}, "float32");
    Constant const_70(const_70_tensor);
    Tensor const_71_tensor({768, 768}, "float32");
    Constant const_71(const_71_tensor);
    Tensor const_72_tensor({768, 768}, "float32");
    Constant const_72(const_72_tensor);
    Tensor const_73_tensor({768, 768}, "float32");
    Constant const_73(const_73_tensor);
    Tensor const_74_tensor({768, 768}, "float32");
    Constant const_74(const_74_tensor);
    Tensor const_75_tensor({768, 3072}, "float32");
    Constant const_75(const_75_tensor);
    Tensor const_76_tensor({3072, 768}, "float32");
    Constant const_76(const_76_tensor);
    Tensor const_77_tensor({768, 768}, "float32");
    Constant const_77(const_77_tensor);
    Tensor const_78_tensor({768, 768}, "float32");
    Constant const_78(const_78_tensor);
    Tensor const_79_tensor({768, 768}, "float32");
    Constant const_79(const_79_tensor);
    Tensor const_80_tensor({768, 768}, "float32");
    Constant const_80(const_80_tensor);
    Tensor const_81_tensor({768, 3072}, "float32");
    Constant const_81(const_81_tensor);
    Tensor const_82_tensor({3072, 768}, "float32");
    Constant const_82(const_82_tensor);
    Tensor const_83_tensor({768, 768}, "float32");
    Constant const_83(const_83_tensor);
    Tensor const_84_tensor({768, 768}, "float32");
    Constant const_84(const_84_tensor);
    Tensor const_85_tensor({768, 768}, "float32");
    Constant const_85(const_85_tensor);
    Tensor const_86_tensor({768, 768}, "float32");
    Constant const_86(const_86_tensor);
    Tensor const_87_tensor({768, 3072}, "float32");
    Constant const_87(const_87_tensor);
    Tensor const_88_tensor({3072, 768}, "float32");
    Constant const_88(const_88_tensor);
    Tensor const_89_tensor({768, 768}, "float32");
    Constant const_89(const_89_tensor);
    Tensor const_90_tensor({768, 768}, "float32");
    Constant const_90(const_90_tensor);
    Tensor const_91_tensor({768, 768}, "float32");
    Constant const_91(const_91_tensor);
    Tensor const_92_tensor({768, 768}, "float32");
    Constant const_92(const_92_tensor);
    Tensor const_93_tensor({768, 3072}, "float32");
    Constant const_93(const_93_tensor);
    Tensor const_94_tensor({3072, 768}, "float32");
    Constant const_94(const_94_tensor);
    Tensor const_95_tensor({768, 768}, "float32");
    Constant const_95(const_95_tensor);
    Tensor const_96_tensor({768, 768}, "float32");
    Constant const_96(const_96_tensor);
    Tensor const_97_tensor({768, 768}, "float32");
    Constant const_97(const_97_tensor);
    Tensor const_98_tensor({768, 768}, "float32");
    Constant const_98(const_98_tensor);
    Tensor const_99_tensor({768, 3072}, "float32");
    Constant const_99(const_99_tensor);
    Tensor const_100_tensor({3072, 768}, "float32");
    Constant const_100(const_100_tensor);
    Tensor const_101_tensor({768, 2}, "float32");
    Constant const_101(const_101_tensor);

    // --- Inputs (Vars) ---
    Var input_102("input_ids");
    Var input_103("attention_mask");

    // --- Operators ---
    // Node: /distilbert/Shape (Shape)
    std::vector<Expr> args_node_104 = {input_102};
    Call node_104(Op::Get("shape"), args_node_104, ObjectRef());
    // Node: /distilbert/Constant (Constant)
    Tensor node_105_tensor({}, "int64");
    Constant node_105(node_105_tensor);
    // Node: /distilbert/Gather (Gather)
    std::vector<Expr> args_node_106 = {node_104, node_105};
    Call node_106(Op::Get("gather"), args_node_106, GatherAttrs::Create(0));
    // Node: /distilbert/embeddings/word_embeddings/Gather (Gather)
    std::vector<Expr> args_node_107 = {const_0, input_102};
    Call node_107(Op::Get("gather"), args_node_107, GatherAttrs::Create(0));
    // Node: /distilbert/embeddings/Shape (Shape)
    std::vector<Expr> args_node_108 = {node_107};
    Call node_108(Op::Get("shape"), args_node_108, ObjectRef());
    // Node: /distilbert/embeddings/Constant (Constant)
    Tensor node_109_tensor({}, "int64");
    Constant node_109(node_109_tensor);
    // Node: /distilbert/embeddings/Gather (Gather)
    std::vector<Expr> args_node_110 = {node_108, node_109};
    Call node_110(Op::Get("gather"), args_node_110, GatherAttrs::Create(0));
    // Node: Constant_187 (Constant)
    Tensor node_111_tensor({1, 512}, "int64");
    Constant node_111(node_111_tensor);
    // Node: /distilbert/embeddings/Constant_1 (Constant)
    Tensor node_112_tensor({1}, "int64");
    Constant node_112(node_112_tensor);
    // Node: /distilbert/embeddings/Constant_2 (Constant)
    Tensor node_113_tensor({1}, "int64");
    Constant node_113(node_113_tensor);
    // Node: /distilbert/embeddings/Constant_3 (Constant)
    Tensor node_114_tensor({1}, "int64");
    Constant node_114(node_114_tensor);
    // Node: /distilbert/embeddings/Unsqueeze (Unsqueeze)
    std::vector<Expr> args_node_115 = {node_110, node_114};
    Call node_115(Op::Get("unsqueeze"), args_node_115, ObjectRef());
    // Node: /distilbert/embeddings/Constant_4 (Constant)
    Tensor node_116_tensor({1}, "int64");
    Constant node_116(node_116_tensor);
    // Node: /distilbert/embeddings/Slice (Slice)
    std::vector<Expr> args_node_117 = {node_111, node_113, node_115, node_112, node_116};
    Call node_117(Op::Get("slice"), args_node_117, ObjectRef());
    // Node: /distilbert/embeddings/position_embeddings/Gather (Gather)
    std::vector<Expr> args_node_118 = {const_1, node_117};
    Call node_118(Op::Get("gather"), args_node_118, GatherAttrs::Create(0));
    // Node: /distilbert/embeddings/Add (Add)
    std::vector<Expr> args_node_119 = {node_107, node_118};
    Call node_119(Op::Get("add"), args_node_119, ObjectRef());
    // Node: /distilbert/embeddings/LayerNorm/ReduceMean (ReduceMean)
    std::vector<Expr> args_node_120 = {node_119};
    Call node_120(Op::Get("reduce_mean"), args_node_120, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/embeddings/LayerNorm/Sub (Sub)
    std::vector<Expr> args_node_121 = {node_119, node_120};
    Call node_121(Op::Get("sub"), args_node_121, ObjectRef());
    // Node: /distilbert/embeddings/LayerNorm/Constant (Constant)
    Tensor node_122_tensor({}, "float32");
    Constant node_122(node_122_tensor);
    // Node: /distilbert/embeddings/LayerNorm/Pow (Pow)
    std::vector<Expr> args_node_123 = {node_121, node_122};
    Call node_123(Op::Get("pow"), args_node_123, ObjectRef());
    // Node: /distilbert/embeddings/LayerNorm/ReduceMean_1 (ReduceMean)
    std::vector<Expr> args_node_124 = {node_123};
    Call node_124(Op::Get("reduce_mean"), args_node_124, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/embeddings/LayerNorm/Constant_1 (Constant)
    Tensor node_125_tensor({}, "float32");
    Constant node_125(node_125_tensor);
    // Node: /distilbert/embeddings/LayerNorm/Add (Add)
    std::vector<Expr> args_node_126 = {node_124, node_125};
    Call node_126(Op::Get("add"), args_node_126, ObjectRef());
    // Node: /distilbert/embeddings/LayerNorm/Sqrt (Sqrt)
    std::vector<Expr> args_node_127 = {node_126};
    Call node_127(Op::Get("sqrt"), args_node_127, ObjectRef());
    // Node: /distilbert/embeddings/LayerNorm/Div (Div)
    std::vector<Expr> args_node_128 = {node_121, node_127};
    Call node_128(Op::Get("divide"), args_node_128, DivAttrs::Create());
    // Node: /distilbert/embeddings/LayerNorm/Mul (Mul)
    std::vector<Expr> args_node_129 = {node_128, const_2};
    Call node_129(Op::Get("mul"), args_node_129, ObjectRef());
    // Node: /distilbert/embeddings/LayerNorm/Add_1 (Add)
    std::vector<Expr> args_node_130 = {node_129, const_3};
    Call node_130(Op::Get("add"), args_node_130, ObjectRef());
    // Node: /distilbert/Shape_1 (Shape)
    std::vector<Expr> args_node_131 = {input_103};
    Call node_131(Op::Get("shape"), args_node_131, ObjectRef());
    // Node: /distilbert/Constant_1 (Constant)
    Tensor node_132_tensor({}, "int64");
    Constant node_132(node_132_tensor);
    // Node: /distilbert/Gather_1 (Gather)
    std::vector<Expr> args_node_133 = {node_131, node_132};
    Call node_133(Op::Get("gather"), args_node_133, GatherAttrs::Create(0));
    // Node: /distilbert/Shape_2 (Shape)
    std::vector<Expr> args_node_134 = {input_103};
    Call node_134(Op::Get("shape"), args_node_134, ObjectRef());
    // Node: /distilbert/Constant_2 (Constant)
    Tensor node_135_tensor({}, "int64");
    Constant node_135(node_135_tensor);
    // Node: /distilbert/Gather_2 (Gather)
    std::vector<Expr> args_node_136 = {node_134, node_135};
    Call node_136(Op::Get("gather"), args_node_136, GatherAttrs::Create(0));
    // Node: /distilbert/Constant_3 (Constant)
    Tensor node_137_tensor({1}, "int64");
    Constant node_137(node_137_tensor);
    // Node: /distilbert/Unsqueeze (Unsqueeze)
    std::vector<Expr> args_node_138 = {input_103, node_137};
    Call node_138(Op::Get("unsqueeze"), args_node_138, ObjectRef());
    // Node: /distilbert/Constant_4 (Constant)
    Tensor node_139_tensor({1}, "int64");
    Constant node_139(node_139_tensor);
    // Node: /distilbert/Unsqueeze_1 (Unsqueeze)
    std::vector<Expr> args_node_140 = {node_138, node_139};
    Call node_140(Op::Get("unsqueeze"), args_node_140, ObjectRef());
    // Node: /distilbert/Constant_5 (Constant)
    Tensor node_141_tensor({1}, "int64");
    Constant node_141(node_141_tensor);
    // Node: /distilbert/Unsqueeze_2 (Unsqueeze)
    std::vector<Expr> args_node_142 = {node_133, node_141};
    Call node_142(Op::Get("unsqueeze"), args_node_142, ObjectRef());
    // Node: /distilbert/Constant_6 (Constant)
    Tensor node_143_tensor({1}, "int64");
    Constant node_143(node_143_tensor);
    // Node: /distilbert/Constant_7 (Constant)
    Tensor node_144_tensor({1}, "int64");
    Constant node_144(node_144_tensor);
    // Node: /distilbert/Unsqueeze_3 (Unsqueeze)
    std::vector<Expr> args_node_145 = {node_106, node_144};
    Call node_145(Op::Get("unsqueeze"), args_node_145, ObjectRef());
    // Node: /distilbert/Constant_8 (Constant)
    Tensor node_146_tensor({1}, "int64");
    Constant node_146(node_146_tensor);
    // Node: /distilbert/Unsqueeze_4 (Unsqueeze)
    std::vector<Expr> args_node_147 = {node_136, node_146};
    Call node_147(Op::Get("unsqueeze"), args_node_147, ObjectRef());
    // Node: /distilbert/Concat (Concat)
    std::vector<Expr> args_node_148 = {node_142, node_143, node_145, node_147};
    Call node_148(Op::Get("concatenate"), args_node_148, ConcatAttrs::Create(0));
    // Node: /distilbert/Constant_9 (Constant)
    Tensor node_149_tensor({1}, "int64");
    Constant node_149(node_149_tensor);
    // Node: /distilbert/Reshape (Reshape)
    std::vector<Expr> args_node_150 = {node_148, node_149};
    Call node_150(Op::Get("reshape"), args_node_150, ReshapeAttrs::Create(0));
    // Node: /distilbert/Shape_3 (Shape)
    std::vector<Expr> args_node_151 = {node_150};
    Call node_151(Op::Get("shape"), args_node_151, ObjectRef());
    Tensor node_152_val({1}, "int64");
    // Node: /distilbert/ConstantOfShape (ConstantOfShape)
    std::vector<Expr> args_node_152 = {node_151};
    Call node_152(Op::Get("constant_of_shape"), args_node_152, ConstantOfShapeAttrs::Create(node_152_val));
    // Node: /distilbert/Constant_10 (Constant)
    Tensor node_153_tensor({}, "int64");
    Constant node_153(node_153_tensor);
    // Node: /distilbert/Mul (Mul)
    std::vector<Expr> args_node_154 = {node_152, node_153};
    Call node_154(Op::Get("mul"), args_node_154, ObjectRef());
    // Node: /distilbert/Equal (Equal)
    std::vector<Expr> args_node_155 = {node_150, node_154};
    Call node_155(Op::Get("equal"), args_node_155, EqualAttrs::Create());
    // Node: /distilbert/Where (Where)
    std::vector<Expr> args_node_156 = {node_155, node_152, node_150};
    Call node_156(Op::Get("where"), args_node_156, ObjectRef());
    // Node: /distilbert/Expand (Expand)
    std::vector<Expr> args_node_157 = {node_140, node_156};
    Call node_157(Op::Get("expand_dims"), args_node_157, ExpandAttrs::Create());
    // Node: /distilbert/Cast (Cast)
    std::vector<Expr> args_node_158 = {node_157};
    Call node_158(Op::Get("cast"), args_node_158, CastAttrs::Create(1));
    // Node: /distilbert/Constant_11 (Constant)
    Tensor node_159_tensor({}, "float32");
    Constant node_159(node_159_tensor);
    // Node: /distilbert/Sub (Sub)
    std::vector<Expr> args_node_160 = {node_159, node_158};
    Call node_160(Op::Get("sub"), args_node_160, ObjectRef());
    // Node: /distilbert/Cast_1 (Cast)
    std::vector<Expr> args_node_161 = {node_160};
    Call node_161(Op::Get("cast"), args_node_161, CastAttrs::Create(9));
    // Node: /distilbert/Cast_2 (Cast)
    std::vector<Expr> args_node_162 = {node_161};
    Call node_162(Op::Get("cast"), args_node_162, CastAttrs::Create(9));
    // Node: /distilbert/Constant_12 (Constant)
    Tensor node_163_tensor({}, "float32");
    Constant node_163(node_163_tensor);
    // Node: /distilbert/Where_1 (Where)
    std::vector<Expr> args_node_164 = {node_162, node_163, node_160};
    Call node_164(Op::Get("where"), args_node_164, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Shape (Shape)
    std::vector<Expr> args_node_165 = {node_130};
    Call node_165(Op::Get("shape"), args_node_165, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Constant (Constant)
    Tensor node_166_tensor({}, "int64");
    Constant node_166(node_166_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Gather (Gather)
    std::vector<Expr> args_node_167 = {node_165, node_166};
    Call node_167(Op::Get("gather"), args_node_167, GatherAttrs::Create(0));
    // Node: /distilbert/transformer/layer.0/attention/q_lin/MatMul (MatMul)
    std::vector<Expr> args_node_168 = {node_130, const_65};
    Call node_168(Op::Get("matmul"), args_node_168, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/q_lin/Add (Add)
    std::vector<Expr> args_node_169 = {const_4, node_168};
    Call node_169(Op::Get("add"), args_node_169, ObjectRef());
    // Node: Constant_246 (Constant)
    Tensor node_170_tensor({1}, "int64");
    Constant node_170(node_170_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Unsqueeze (Unsqueeze)
    std::vector<Expr> args_node_171 = {node_167, node_170};
    Call node_171(Op::Get("unsqueeze"), args_node_171, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Constant_1 (Constant)
    Tensor node_172_tensor({1}, "int64");
    Constant node_172(node_172_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Constant_2 (Constant)
    Tensor node_173_tensor({1}, "int64");
    Constant node_173(node_173_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Constant_3 (Constant)
    Tensor node_174_tensor({1}, "int64");
    Constant node_174(node_174_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Concat (Concat)
    std::vector<Expr> args_node_175 = {node_171, node_172, node_173, node_174};
    Call node_175(Op::Get("concatenate"), args_node_175, ConcatAttrs::Create(0));
    // Node: Constant_252 (Constant)
    Tensor node_176_tensor({1}, "int64");
    Constant node_176(node_176_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Unsqueeze_1 (Unsqueeze)
    std::vector<Expr> args_node_177 = {node_167, node_176};
    Call node_177(Op::Get("unsqueeze"), args_node_177, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Constant_4 (Constant)
    Tensor node_178_tensor({1}, "int64");
    Constant node_178(node_178_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Constant_5 (Constant)
    Tensor node_179_tensor({1}, "int64");
    Constant node_179(node_179_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Constant_6 (Constant)
    Tensor node_180_tensor({1}, "int64");
    Constant node_180(node_180_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Concat_1 (Concat)
    std::vector<Expr> args_node_181 = {node_177, node_178, node_179, node_180};
    Call node_181(Op::Get("concatenate"), args_node_181, ConcatAttrs::Create(0));
    // Node: Constant_258 (Constant)
    Tensor node_182_tensor({1}, "int64");
    Constant node_182(node_182_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Unsqueeze_2 (Unsqueeze)
    std::vector<Expr> args_node_183 = {node_167, node_182};
    Call node_183(Op::Get("unsqueeze"), args_node_183, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Constant_7 (Constant)
    Tensor node_184_tensor({1}, "int64");
    Constant node_184(node_184_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Constant_8 (Constant)
    Tensor node_185_tensor({1}, "int64");
    Constant node_185(node_185_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Constant_9 (Constant)
    Tensor node_186_tensor({1}, "int64");
    Constant node_186(node_186_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Concat_2 (Concat)
    std::vector<Expr> args_node_187 = {node_183, node_184, node_185, node_186};
    Call node_187(Op::Get("concatenate"), args_node_187, ConcatAttrs::Create(0));
    // Node: /distilbert/transformer/layer.0/attention/Reshape (Reshape)
    std::vector<Expr> args_node_188 = {node_169, node_175};
    Call node_188(Op::Get("reshape"), args_node_188, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.0/attention/Transpose (Transpose)
    std::vector<Expr> args_node_189 = {node_188};
    Call node_189(Op::Get("transpose"), args_node_189, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: /distilbert/transformer/layer.0/attention/k_lin/MatMul (MatMul)
    std::vector<Expr> args_node_190 = {node_130, const_66};
    Call node_190(Op::Get("matmul"), args_node_190, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/k_lin/Add (Add)
    std::vector<Expr> args_node_191 = {const_5, node_190};
    Call node_191(Op::Get("add"), args_node_191, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Reshape_1 (Reshape)
    std::vector<Expr> args_node_192 = {node_191, node_181};
    Call node_192(Op::Get("reshape"), args_node_192, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.0/attention/v_lin/MatMul (MatMul)
    std::vector<Expr> args_node_193 = {node_130, const_67};
    Call node_193(Op::Get("matmul"), args_node_193, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/v_lin/Add (Add)
    std::vector<Expr> args_node_194 = {const_6, node_193};
    Call node_194(Op::Get("add"), args_node_194, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Reshape_2 (Reshape)
    std::vector<Expr> args_node_195 = {node_194, node_187};
    Call node_195(Op::Get("reshape"), args_node_195, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.0/attention/Transpose_1 (Transpose)
    std::vector<Expr> args_node_196 = {node_195};
    Call node_196(Op::Get("transpose"), args_node_196, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: /distilbert/transformer/layer.0/attention/Shape_1 (Shape)
    std::vector<Expr> args_node_197 = {node_189};
    Call node_197(Op::Get("shape"), args_node_197, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Constant_10 (Constant)
    Tensor node_198_tensor({1}, "int64");
    Constant node_198(node_198_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Constant_11 (Constant)
    Tensor node_199_tensor({1}, "int64");
    Constant node_199(node_199_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Slice (Slice)
    std::vector<Expr> args_node_200 = {node_197, node_198, node_199};
    Call node_200(Op::Get("slice"), args_node_200, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Cast (Cast)
    std::vector<Expr> args_node_201 = {node_200};
    Call node_201(Op::Get("cast"), args_node_201, CastAttrs::Create(1));
    // Node: /distilbert/transformer/layer.0/attention/Sqrt (Sqrt)
    std::vector<Expr> args_node_202 = {node_201};
    Call node_202(Op::Get("sqrt"), args_node_202, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Constant_12 (Constant)
    Tensor node_203_tensor({1}, "float32");
    Constant node_203(node_203_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Div (Div)
    std::vector<Expr> args_node_204 = {node_203, node_202};
    Call node_204(Op::Get("divide"), args_node_204, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.0/attention/Cast_1 (Cast)
    std::vector<Expr> args_node_205 = {node_204};
    Call node_205(Op::Get("cast"), args_node_205, CastAttrs::Create(1));
    // Node: /distilbert/transformer/layer.0/attention/Transpose_2 (Transpose)
    std::vector<Expr> args_node_206 = {node_192};
    Call node_206(Op::Get("transpose"), args_node_206, TransposeAttrs::Create({0, 2, 3, 1}));
    // Node: /distilbert/transformer/layer.0/attention/Sqrt_1 (Sqrt)
    std::vector<Expr> args_node_207 = {node_205};
    Call node_207(Op::Get("sqrt"), args_node_207, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Mul (Mul)
    std::vector<Expr> args_node_208 = {node_189, node_207};
    Call node_208(Op::Get("mul"), args_node_208, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Sqrt_2 (Sqrt)
    std::vector<Expr> args_node_209 = {node_205};
    Call node_209(Op::Get("sqrt"), args_node_209, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Mul_1 (Mul)
    std::vector<Expr> args_node_210 = {node_206, node_209};
    Call node_210(Op::Get("mul"), args_node_210, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/MatMul (MatMul)
    std::vector<Expr> args_node_211 = {node_208, node_210};
    Call node_211(Op::Get("matmul"), args_node_211, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Add (Add)
    std::vector<Expr> args_node_212 = {node_211, node_164};
    Call node_212(Op::Get("add"), args_node_212, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Softmax (Softmax)
    std::vector<Expr> args_node_213 = {node_212};
    Call node_213(Op::Get("softmax"), args_node_213, SoftmaxAttrs::Create(-1));
    // Node: /distilbert/transformer/layer.0/attention/MatMul_1 (MatMul)
    std::vector<Expr> args_node_214 = {node_213, node_196};
    Call node_214(Op::Get("matmul"), args_node_214, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Transpose_3 (Transpose)
    std::vector<Expr> args_node_215 = {node_214};
    Call node_215(Op::Get("transpose"), args_node_215, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: Constant_292 (Constant)
    Tensor node_216_tensor({1}, "int64");
    Constant node_216(node_216_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Unsqueeze_3 (Unsqueeze)
    std::vector<Expr> args_node_217 = {node_167, node_216};
    Call node_217(Op::Get("unsqueeze"), args_node_217, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/Constant_13 (Constant)
    Tensor node_218_tensor({1}, "int64");
    Constant node_218(node_218_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Constant_14 (Constant)
    Tensor node_219_tensor({1}, "int64");
    Constant node_219(node_219_tensor);
    // Node: /distilbert/transformer/layer.0/attention/Concat_3 (Concat)
    std::vector<Expr> args_node_220 = {node_217, node_218, node_219};
    Call node_220(Op::Get("concatenate"), args_node_220, ConcatAttrs::Create(0));
    // Node: /distilbert/transformer/layer.0/attention/Reshape_3 (Reshape)
    std::vector<Expr> args_node_221 = {node_215, node_220};
    Call node_221(Op::Get("reshape"), args_node_221, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.0/attention/out_lin/MatMul (MatMul)
    std::vector<Expr> args_node_222 = {node_221, const_68};
    Call node_222(Op::Get("matmul"), args_node_222, ObjectRef());
    // Node: /distilbert/transformer/layer.0/attention/out_lin/Add (Add)
    std::vector<Expr> args_node_223 = {const_7, node_222};
    Call node_223(Op::Get("add"), args_node_223, ObjectRef());
    // Node: /distilbert/transformer/layer.0/Add (Add)
    std::vector<Expr> args_node_224 = {node_223, node_130};
    Call node_224(Op::Get("add"), args_node_224, ObjectRef());
    // Node: /distilbert/transformer/layer.0/sa_layer_norm/ReduceMean (ReduceMean)
    std::vector<Expr> args_node_225 = {node_224};
    Call node_225(Op::Get("reduce_mean"), args_node_225, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.0/sa_layer_norm/Sub (Sub)
    std::vector<Expr> args_node_226 = {node_224, node_225};
    Call node_226(Op::Get("sub"), args_node_226, ObjectRef());
    // Node: /distilbert/transformer/layer.0/sa_layer_norm/Constant (Constant)
    Tensor node_227_tensor({}, "float32");
    Constant node_227(node_227_tensor);
    // Node: /distilbert/transformer/layer.0/sa_layer_norm/Pow (Pow)
    std::vector<Expr> args_node_228 = {node_226, node_227};
    Call node_228(Op::Get("pow"), args_node_228, ObjectRef());
    // Node: /distilbert/transformer/layer.0/sa_layer_norm/ReduceMean_1 (ReduceMean)
    std::vector<Expr> args_node_229 = {node_228};
    Call node_229(Op::Get("reduce_mean"), args_node_229, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.0/sa_layer_norm/Constant_1 (Constant)
    Tensor node_230_tensor({}, "float32");
    Constant node_230(node_230_tensor);
    // Node: /distilbert/transformer/layer.0/sa_layer_norm/Add (Add)
    std::vector<Expr> args_node_231 = {node_229, node_230};
    Call node_231(Op::Get("add"), args_node_231, ObjectRef());
    // Node: /distilbert/transformer/layer.0/sa_layer_norm/Sqrt (Sqrt)
    std::vector<Expr> args_node_232 = {node_231};
    Call node_232(Op::Get("sqrt"), args_node_232, ObjectRef());
    // Node: /distilbert/transformer/layer.0/sa_layer_norm/Div (Div)
    std::vector<Expr> args_node_233 = {node_226, node_232};
    Call node_233(Op::Get("divide"), args_node_233, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.0/sa_layer_norm/Mul (Mul)
    std::vector<Expr> args_node_234 = {node_233, const_8};
    Call node_234(Op::Get("mul"), args_node_234, ObjectRef());
    // Node: /distilbert/transformer/layer.0/sa_layer_norm/Add_1 (Add)
    std::vector<Expr> args_node_235 = {node_234, const_9};
    Call node_235(Op::Get("add"), args_node_235, ObjectRef());
    // Node: /distilbert/transformer/layer.0/ffn/lin1/MatMul (MatMul)
    std::vector<Expr> args_node_236 = {node_235, const_69};
    Call node_236(Op::Get("matmul"), args_node_236, ObjectRef());
    // Node: /distilbert/transformer/layer.0/ffn/lin1/Add (Add)
    std::vector<Expr> args_node_237 = {const_10, node_236};
    Call node_237(Op::Get("add"), args_node_237, ObjectRef());
    // Node: /distilbert/transformer/layer.0/ffn/activation/Constant (Constant)
    Tensor node_238_tensor({}, "float32");
    Constant node_238(node_238_tensor);
    // Node: /distilbert/transformer/layer.0/ffn/activation/Div (Div)
    std::vector<Expr> args_node_239 = {node_237, node_238};
    Call node_239(Op::Get("divide"), args_node_239, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.0/ffn/activation/Erf (Erf)
    std::vector<Expr> args_node_240 = {node_239};
    Call node_240(Op::Get("erf"), args_node_240, ErfAttrs::Create());
    // Node: /distilbert/transformer/layer.0/ffn/activation/Constant_1 (Constant)
    Tensor node_241_tensor({}, "float32");
    Constant node_241(node_241_tensor);
    // Node: /distilbert/transformer/layer.0/ffn/activation/Add (Add)
    std::vector<Expr> args_node_242 = {node_240, node_241};
    Call node_242(Op::Get("add"), args_node_242, ObjectRef());
    // Node: /distilbert/transformer/layer.0/ffn/activation/Mul (Mul)
    std::vector<Expr> args_node_243 = {node_237, node_242};
    Call node_243(Op::Get("mul"), args_node_243, ObjectRef());
    // Node: /distilbert/transformer/layer.0/ffn/activation/Constant_2 (Constant)
    Tensor node_244_tensor({}, "float32");
    Constant node_244(node_244_tensor);
    // Node: /distilbert/transformer/layer.0/ffn/activation/Mul_1 (Mul)
    std::vector<Expr> args_node_245 = {node_243, node_244};
    Call node_245(Op::Get("mul"), args_node_245, ObjectRef());
    // Node: /distilbert/transformer/layer.0/ffn/lin2/MatMul (MatMul)
    std::vector<Expr> args_node_246 = {node_245, const_70};
    Call node_246(Op::Get("matmul"), args_node_246, ObjectRef());
    // Node: /distilbert/transformer/layer.0/ffn/lin2/Add (Add)
    std::vector<Expr> args_node_247 = {const_11, node_246};
    Call node_247(Op::Get("add"), args_node_247, ObjectRef());
    // Node: /distilbert/transformer/layer.0/Add_1 (Add)
    std::vector<Expr> args_node_248 = {node_247, node_235};
    Call node_248(Op::Get("add"), args_node_248, ObjectRef());
    // Node: /distilbert/transformer/layer.0/output_layer_norm/ReduceMean (ReduceMean)
    std::vector<Expr> args_node_249 = {node_248};
    Call node_249(Op::Get("reduce_mean"), args_node_249, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.0/output_layer_norm/Sub (Sub)
    std::vector<Expr> args_node_250 = {node_248, node_249};
    Call node_250(Op::Get("sub"), args_node_250, ObjectRef());
    // Node: /distilbert/transformer/layer.0/output_layer_norm/Constant (Constant)
    Tensor node_251_tensor({}, "float32");
    Constant node_251(node_251_tensor);
    // Node: /distilbert/transformer/layer.0/output_layer_norm/Pow (Pow)
    std::vector<Expr> args_node_252 = {node_250, node_251};
    Call node_252(Op::Get("pow"), args_node_252, ObjectRef());
    // Node: /distilbert/transformer/layer.0/output_layer_norm/ReduceMean_1 (ReduceMean)
    std::vector<Expr> args_node_253 = {node_252};
    Call node_253(Op::Get("reduce_mean"), args_node_253, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.0/output_layer_norm/Constant_1 (Constant)
    Tensor node_254_tensor({}, "float32");
    Constant node_254(node_254_tensor);
    // Node: /distilbert/transformer/layer.0/output_layer_norm/Add (Add)
    std::vector<Expr> args_node_255 = {node_253, node_254};
    Call node_255(Op::Get("add"), args_node_255, ObjectRef());
    // Node: /distilbert/transformer/layer.0/output_layer_norm/Sqrt (Sqrt)
    std::vector<Expr> args_node_256 = {node_255};
    Call node_256(Op::Get("sqrt"), args_node_256, ObjectRef());
    // Node: /distilbert/transformer/layer.0/output_layer_norm/Div (Div)
    std::vector<Expr> args_node_257 = {node_250, node_256};
    Call node_257(Op::Get("divide"), args_node_257, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.0/output_layer_norm/Mul (Mul)
    std::vector<Expr> args_node_258 = {node_257, const_12};
    Call node_258(Op::Get("mul"), args_node_258, ObjectRef());
    // Node: /distilbert/transformer/layer.0/output_layer_norm/Add_1 (Add)
    std::vector<Expr> args_node_259 = {node_258, const_13};
    Call node_259(Op::Get("add"), args_node_259, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Shape (Shape)
    std::vector<Expr> args_node_260 = {node_259};
    Call node_260(Op::Get("shape"), args_node_260, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Constant (Constant)
    Tensor node_261_tensor({}, "int64");
    Constant node_261(node_261_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Gather (Gather)
    std::vector<Expr> args_node_262 = {node_260, node_261};
    Call node_262(Op::Get("gather"), args_node_262, GatherAttrs::Create(0));
    // Node: /distilbert/transformer/layer.1/attention/q_lin/MatMul (MatMul)
    std::vector<Expr> args_node_263 = {node_259, const_71};
    Call node_263(Op::Get("matmul"), args_node_263, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/q_lin/Add (Add)
    std::vector<Expr> args_node_264 = {const_14, node_263};
    Call node_264(Op::Get("add"), args_node_264, ObjectRef());
    // Node: Constant_341 (Constant)
    Tensor node_265_tensor({1}, "int64");
    Constant node_265(node_265_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Unsqueeze (Unsqueeze)
    std::vector<Expr> args_node_266 = {node_262, node_265};
    Call node_266(Op::Get("unsqueeze"), args_node_266, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Constant_1 (Constant)
    Tensor node_267_tensor({1}, "int64");
    Constant node_267(node_267_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Constant_2 (Constant)
    Tensor node_268_tensor({1}, "int64");
    Constant node_268(node_268_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Constant_3 (Constant)
    Tensor node_269_tensor({1}, "int64");
    Constant node_269(node_269_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Concat (Concat)
    std::vector<Expr> args_node_270 = {node_266, node_267, node_268, node_269};
    Call node_270(Op::Get("concatenate"), args_node_270, ConcatAttrs::Create(0));
    // Node: Constant_347 (Constant)
    Tensor node_271_tensor({1}, "int64");
    Constant node_271(node_271_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Unsqueeze_1 (Unsqueeze)
    std::vector<Expr> args_node_272 = {node_262, node_271};
    Call node_272(Op::Get("unsqueeze"), args_node_272, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Constant_4 (Constant)
    Tensor node_273_tensor({1}, "int64");
    Constant node_273(node_273_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Constant_5 (Constant)
    Tensor node_274_tensor({1}, "int64");
    Constant node_274(node_274_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Constant_6 (Constant)
    Tensor node_275_tensor({1}, "int64");
    Constant node_275(node_275_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Concat_1 (Concat)
    std::vector<Expr> args_node_276 = {node_272, node_273, node_274, node_275};
    Call node_276(Op::Get("concatenate"), args_node_276, ConcatAttrs::Create(0));
    // Node: Constant_353 (Constant)
    Tensor node_277_tensor({1}, "int64");
    Constant node_277(node_277_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Unsqueeze_2 (Unsqueeze)
    std::vector<Expr> args_node_278 = {node_262, node_277};
    Call node_278(Op::Get("unsqueeze"), args_node_278, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Constant_7 (Constant)
    Tensor node_279_tensor({1}, "int64");
    Constant node_279(node_279_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Constant_8 (Constant)
    Tensor node_280_tensor({1}, "int64");
    Constant node_280(node_280_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Constant_9 (Constant)
    Tensor node_281_tensor({1}, "int64");
    Constant node_281(node_281_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Concat_2 (Concat)
    std::vector<Expr> args_node_282 = {node_278, node_279, node_280, node_281};
    Call node_282(Op::Get("concatenate"), args_node_282, ConcatAttrs::Create(0));
    // Node: /distilbert/transformer/layer.1/attention/Reshape (Reshape)
    std::vector<Expr> args_node_283 = {node_264, node_270};
    Call node_283(Op::Get("reshape"), args_node_283, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.1/attention/Transpose (Transpose)
    std::vector<Expr> args_node_284 = {node_283};
    Call node_284(Op::Get("transpose"), args_node_284, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: /distilbert/transformer/layer.1/attention/k_lin/MatMul (MatMul)
    std::vector<Expr> args_node_285 = {node_259, const_72};
    Call node_285(Op::Get("matmul"), args_node_285, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/k_lin/Add (Add)
    std::vector<Expr> args_node_286 = {const_15, node_285};
    Call node_286(Op::Get("add"), args_node_286, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Reshape_1 (Reshape)
    std::vector<Expr> args_node_287 = {node_286, node_276};
    Call node_287(Op::Get("reshape"), args_node_287, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.1/attention/v_lin/MatMul (MatMul)
    std::vector<Expr> args_node_288 = {node_259, const_73};
    Call node_288(Op::Get("matmul"), args_node_288, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/v_lin/Add (Add)
    std::vector<Expr> args_node_289 = {const_16, node_288};
    Call node_289(Op::Get("add"), args_node_289, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Reshape_2 (Reshape)
    std::vector<Expr> args_node_290 = {node_289, node_282};
    Call node_290(Op::Get("reshape"), args_node_290, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.1/attention/Transpose_1 (Transpose)
    std::vector<Expr> args_node_291 = {node_290};
    Call node_291(Op::Get("transpose"), args_node_291, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: /distilbert/transformer/layer.1/attention/Shape_1 (Shape)
    std::vector<Expr> args_node_292 = {node_284};
    Call node_292(Op::Get("shape"), args_node_292, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Constant_10 (Constant)
    Tensor node_293_tensor({1}, "int64");
    Constant node_293(node_293_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Constant_11 (Constant)
    Tensor node_294_tensor({1}, "int64");
    Constant node_294(node_294_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Slice (Slice)
    std::vector<Expr> args_node_295 = {node_292, node_293, node_294};
    Call node_295(Op::Get("slice"), args_node_295, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Cast (Cast)
    std::vector<Expr> args_node_296 = {node_295};
    Call node_296(Op::Get("cast"), args_node_296, CastAttrs::Create(1));
    // Node: /distilbert/transformer/layer.1/attention/Sqrt (Sqrt)
    std::vector<Expr> args_node_297 = {node_296};
    Call node_297(Op::Get("sqrt"), args_node_297, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Constant_12 (Constant)
    Tensor node_298_tensor({1}, "float32");
    Constant node_298(node_298_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Div (Div)
    std::vector<Expr> args_node_299 = {node_298, node_297};
    Call node_299(Op::Get("divide"), args_node_299, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.1/attention/Cast_1 (Cast)
    std::vector<Expr> args_node_300 = {node_299};
    Call node_300(Op::Get("cast"), args_node_300, CastAttrs::Create(1));
    // Node: /distilbert/transformer/layer.1/attention/Transpose_2 (Transpose)
    std::vector<Expr> args_node_301 = {node_287};
    Call node_301(Op::Get("transpose"), args_node_301, TransposeAttrs::Create({0, 2, 3, 1}));
    // Node: /distilbert/transformer/layer.1/attention/Sqrt_1 (Sqrt)
    std::vector<Expr> args_node_302 = {node_300};
    Call node_302(Op::Get("sqrt"), args_node_302, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Mul (Mul)
    std::vector<Expr> args_node_303 = {node_284, node_302};
    Call node_303(Op::Get("mul"), args_node_303, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Sqrt_2 (Sqrt)
    std::vector<Expr> args_node_304 = {node_300};
    Call node_304(Op::Get("sqrt"), args_node_304, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Mul_1 (Mul)
    std::vector<Expr> args_node_305 = {node_301, node_304};
    Call node_305(Op::Get("mul"), args_node_305, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/MatMul (MatMul)
    std::vector<Expr> args_node_306 = {node_303, node_305};
    Call node_306(Op::Get("matmul"), args_node_306, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Add (Add)
    std::vector<Expr> args_node_307 = {node_306, node_164};
    Call node_307(Op::Get("add"), args_node_307, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Softmax (Softmax)
    std::vector<Expr> args_node_308 = {node_307};
    Call node_308(Op::Get("softmax"), args_node_308, SoftmaxAttrs::Create(-1));
    // Node: /distilbert/transformer/layer.1/attention/MatMul_1 (MatMul)
    std::vector<Expr> args_node_309 = {node_308, node_291};
    Call node_309(Op::Get("matmul"), args_node_309, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Transpose_3 (Transpose)
    std::vector<Expr> args_node_310 = {node_309};
    Call node_310(Op::Get("transpose"), args_node_310, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: Constant_387 (Constant)
    Tensor node_311_tensor({1}, "int64");
    Constant node_311(node_311_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Unsqueeze_3 (Unsqueeze)
    std::vector<Expr> args_node_312 = {node_262, node_311};
    Call node_312(Op::Get("unsqueeze"), args_node_312, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/Constant_13 (Constant)
    Tensor node_313_tensor({1}, "int64");
    Constant node_313(node_313_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Constant_14 (Constant)
    Tensor node_314_tensor({1}, "int64");
    Constant node_314(node_314_tensor);
    // Node: /distilbert/transformer/layer.1/attention/Concat_3 (Concat)
    std::vector<Expr> args_node_315 = {node_312, node_313, node_314};
    Call node_315(Op::Get("concatenate"), args_node_315, ConcatAttrs::Create(0));
    // Node: /distilbert/transformer/layer.1/attention/Reshape_3 (Reshape)
    std::vector<Expr> args_node_316 = {node_310, node_315};
    Call node_316(Op::Get("reshape"), args_node_316, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.1/attention/out_lin/MatMul (MatMul)
    std::vector<Expr> args_node_317 = {node_316, const_74};
    Call node_317(Op::Get("matmul"), args_node_317, ObjectRef());
    // Node: /distilbert/transformer/layer.1/attention/out_lin/Add (Add)
    std::vector<Expr> args_node_318 = {const_17, node_317};
    Call node_318(Op::Get("add"), args_node_318, ObjectRef());
    // Node: /distilbert/transformer/layer.1/Add (Add)
    std::vector<Expr> args_node_319 = {node_318, node_259};
    Call node_319(Op::Get("add"), args_node_319, ObjectRef());
    // Node: /distilbert/transformer/layer.1/sa_layer_norm/ReduceMean (ReduceMean)
    std::vector<Expr> args_node_320 = {node_319};
    Call node_320(Op::Get("reduce_mean"), args_node_320, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.1/sa_layer_norm/Sub (Sub)
    std::vector<Expr> args_node_321 = {node_319, node_320};
    Call node_321(Op::Get("sub"), args_node_321, ObjectRef());
    // Node: /distilbert/transformer/layer.1/sa_layer_norm/Constant (Constant)
    Tensor node_322_tensor({}, "float32");
    Constant node_322(node_322_tensor);
    // Node: /distilbert/transformer/layer.1/sa_layer_norm/Pow (Pow)
    std::vector<Expr> args_node_323 = {node_321, node_322};
    Call node_323(Op::Get("pow"), args_node_323, ObjectRef());
    // Node: /distilbert/transformer/layer.1/sa_layer_norm/ReduceMean_1 (ReduceMean)
    std::vector<Expr> args_node_324 = {node_323};
    Call node_324(Op::Get("reduce_mean"), args_node_324, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.1/sa_layer_norm/Constant_1 (Constant)
    Tensor node_325_tensor({}, "float32");
    Constant node_325(node_325_tensor);
    // Node: /distilbert/transformer/layer.1/sa_layer_norm/Add (Add)
    std::vector<Expr> args_node_326 = {node_324, node_325};
    Call node_326(Op::Get("add"), args_node_326, ObjectRef());
    // Node: /distilbert/transformer/layer.1/sa_layer_norm/Sqrt (Sqrt)
    std::vector<Expr> args_node_327 = {node_326};
    Call node_327(Op::Get("sqrt"), args_node_327, ObjectRef());
    // Node: /distilbert/transformer/layer.1/sa_layer_norm/Div (Div)
    std::vector<Expr> args_node_328 = {node_321, node_327};
    Call node_328(Op::Get("divide"), args_node_328, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.1/sa_layer_norm/Mul (Mul)
    std::vector<Expr> args_node_329 = {node_328, const_18};
    Call node_329(Op::Get("mul"), args_node_329, ObjectRef());
    // Node: /distilbert/transformer/layer.1/sa_layer_norm/Add_1 (Add)
    std::vector<Expr> args_node_330 = {node_329, const_19};
    Call node_330(Op::Get("add"), args_node_330, ObjectRef());
    // Node: /distilbert/transformer/layer.1/ffn/lin1/MatMul (MatMul)
    std::vector<Expr> args_node_331 = {node_330, const_75};
    Call node_331(Op::Get("matmul"), args_node_331, ObjectRef());
    // Node: /distilbert/transformer/layer.1/ffn/lin1/Add (Add)
    std::vector<Expr> args_node_332 = {const_20, node_331};
    Call node_332(Op::Get("add"), args_node_332, ObjectRef());
    // Node: /distilbert/transformer/layer.1/ffn/activation/Constant (Constant)
    Tensor node_333_tensor({}, "float32");
    Constant node_333(node_333_tensor);
    // Node: /distilbert/transformer/layer.1/ffn/activation/Div (Div)
    std::vector<Expr> args_node_334 = {node_332, node_333};
    Call node_334(Op::Get("divide"), args_node_334, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.1/ffn/activation/Erf (Erf)
    std::vector<Expr> args_node_335 = {node_334};
    Call node_335(Op::Get("erf"), args_node_335, ErfAttrs::Create());
    // Node: /distilbert/transformer/layer.1/ffn/activation/Constant_1 (Constant)
    Tensor node_336_tensor({}, "float32");
    Constant node_336(node_336_tensor);
    // Node: /distilbert/transformer/layer.1/ffn/activation/Add (Add)
    std::vector<Expr> args_node_337 = {node_335, node_336};
    Call node_337(Op::Get("add"), args_node_337, ObjectRef());
    // Node: /distilbert/transformer/layer.1/ffn/activation/Mul (Mul)
    std::vector<Expr> args_node_338 = {node_332, node_337};
    Call node_338(Op::Get("mul"), args_node_338, ObjectRef());
    // Node: /distilbert/transformer/layer.1/ffn/activation/Constant_2 (Constant)
    Tensor node_339_tensor({}, "float32");
    Constant node_339(node_339_tensor);
    // Node: /distilbert/transformer/layer.1/ffn/activation/Mul_1 (Mul)
    std::vector<Expr> args_node_340 = {node_338, node_339};
    Call node_340(Op::Get("mul"), args_node_340, ObjectRef());
    // Node: /distilbert/transformer/layer.1/ffn/lin2/MatMul (MatMul)
    std::vector<Expr> args_node_341 = {node_340, const_76};
    Call node_341(Op::Get("matmul"), args_node_341, ObjectRef());
    // Node: /distilbert/transformer/layer.1/ffn/lin2/Add (Add)
    std::vector<Expr> args_node_342 = {const_21, node_341};
    Call node_342(Op::Get("add"), args_node_342, ObjectRef());
    // Node: /distilbert/transformer/layer.1/Add_1 (Add)
    std::vector<Expr> args_node_343 = {node_342, node_330};
    Call node_343(Op::Get("add"), args_node_343, ObjectRef());
    // Node: /distilbert/transformer/layer.1/output_layer_norm/ReduceMean (ReduceMean)
    std::vector<Expr> args_node_344 = {node_343};
    Call node_344(Op::Get("reduce_mean"), args_node_344, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.1/output_layer_norm/Sub (Sub)
    std::vector<Expr> args_node_345 = {node_343, node_344};
    Call node_345(Op::Get("sub"), args_node_345, ObjectRef());
    // Node: /distilbert/transformer/layer.1/output_layer_norm/Constant (Constant)
    Tensor node_346_tensor({}, "float32");
    Constant node_346(node_346_tensor);
    // Node: /distilbert/transformer/layer.1/output_layer_norm/Pow (Pow)
    std::vector<Expr> args_node_347 = {node_345, node_346};
    Call node_347(Op::Get("pow"), args_node_347, ObjectRef());
    // Node: /distilbert/transformer/layer.1/output_layer_norm/ReduceMean_1 (ReduceMean)
    std::vector<Expr> args_node_348 = {node_347};
    Call node_348(Op::Get("reduce_mean"), args_node_348, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.1/output_layer_norm/Constant_1 (Constant)
    Tensor node_349_tensor({}, "float32");
    Constant node_349(node_349_tensor);
    // Node: /distilbert/transformer/layer.1/output_layer_norm/Add (Add)
    std::vector<Expr> args_node_350 = {node_348, node_349};
    Call node_350(Op::Get("add"), args_node_350, ObjectRef());
    // Node: /distilbert/transformer/layer.1/output_layer_norm/Sqrt (Sqrt)
    std::vector<Expr> args_node_351 = {node_350};
    Call node_351(Op::Get("sqrt"), args_node_351, ObjectRef());
    // Node: /distilbert/transformer/layer.1/output_layer_norm/Div (Div)
    std::vector<Expr> args_node_352 = {node_345, node_351};
    Call node_352(Op::Get("divide"), args_node_352, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.1/output_layer_norm/Mul (Mul)
    std::vector<Expr> args_node_353 = {node_352, const_22};
    Call node_353(Op::Get("mul"), args_node_353, ObjectRef());
    // Node: /distilbert/transformer/layer.1/output_layer_norm/Add_1 (Add)
    std::vector<Expr> args_node_354 = {node_353, const_23};
    Call node_354(Op::Get("add"), args_node_354, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Shape (Shape)
    std::vector<Expr> args_node_355 = {node_354};
    Call node_355(Op::Get("shape"), args_node_355, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Constant (Constant)
    Tensor node_356_tensor({}, "int64");
    Constant node_356(node_356_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Gather (Gather)
    std::vector<Expr> args_node_357 = {node_355, node_356};
    Call node_357(Op::Get("gather"), args_node_357, GatherAttrs::Create(0));
    // Node: /distilbert/transformer/layer.2/attention/q_lin/MatMul (MatMul)
    std::vector<Expr> args_node_358 = {node_354, const_77};
    Call node_358(Op::Get("matmul"), args_node_358, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/q_lin/Add (Add)
    std::vector<Expr> args_node_359 = {const_24, node_358};
    Call node_359(Op::Get("add"), args_node_359, ObjectRef());
    // Node: Constant_436 (Constant)
    Tensor node_360_tensor({1}, "int64");
    Constant node_360(node_360_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Unsqueeze (Unsqueeze)
    std::vector<Expr> args_node_361 = {node_357, node_360};
    Call node_361(Op::Get("unsqueeze"), args_node_361, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Constant_1 (Constant)
    Tensor node_362_tensor({1}, "int64");
    Constant node_362(node_362_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Constant_2 (Constant)
    Tensor node_363_tensor({1}, "int64");
    Constant node_363(node_363_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Constant_3 (Constant)
    Tensor node_364_tensor({1}, "int64");
    Constant node_364(node_364_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Concat (Concat)
    std::vector<Expr> args_node_365 = {node_361, node_362, node_363, node_364};
    Call node_365(Op::Get("concatenate"), args_node_365, ConcatAttrs::Create(0));
    // Node: Constant_442 (Constant)
    Tensor node_366_tensor({1}, "int64");
    Constant node_366(node_366_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Unsqueeze_1 (Unsqueeze)
    std::vector<Expr> args_node_367 = {node_357, node_366};
    Call node_367(Op::Get("unsqueeze"), args_node_367, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Constant_4 (Constant)
    Tensor node_368_tensor({1}, "int64");
    Constant node_368(node_368_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Constant_5 (Constant)
    Tensor node_369_tensor({1}, "int64");
    Constant node_369(node_369_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Constant_6 (Constant)
    Tensor node_370_tensor({1}, "int64");
    Constant node_370(node_370_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Concat_1 (Concat)
    std::vector<Expr> args_node_371 = {node_367, node_368, node_369, node_370};
    Call node_371(Op::Get("concatenate"), args_node_371, ConcatAttrs::Create(0));
    // Node: Constant_448 (Constant)
    Tensor node_372_tensor({1}, "int64");
    Constant node_372(node_372_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Unsqueeze_2 (Unsqueeze)
    std::vector<Expr> args_node_373 = {node_357, node_372};
    Call node_373(Op::Get("unsqueeze"), args_node_373, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Constant_7 (Constant)
    Tensor node_374_tensor({1}, "int64");
    Constant node_374(node_374_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Constant_8 (Constant)
    Tensor node_375_tensor({1}, "int64");
    Constant node_375(node_375_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Constant_9 (Constant)
    Tensor node_376_tensor({1}, "int64");
    Constant node_376(node_376_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Concat_2 (Concat)
    std::vector<Expr> args_node_377 = {node_373, node_374, node_375, node_376};
    Call node_377(Op::Get("concatenate"), args_node_377, ConcatAttrs::Create(0));
    // Node: /distilbert/transformer/layer.2/attention/Reshape (Reshape)
    std::vector<Expr> args_node_378 = {node_359, node_365};
    Call node_378(Op::Get("reshape"), args_node_378, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.2/attention/Transpose (Transpose)
    std::vector<Expr> args_node_379 = {node_378};
    Call node_379(Op::Get("transpose"), args_node_379, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: /distilbert/transformer/layer.2/attention/k_lin/MatMul (MatMul)
    std::vector<Expr> args_node_380 = {node_354, const_78};
    Call node_380(Op::Get("matmul"), args_node_380, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/k_lin/Add (Add)
    std::vector<Expr> args_node_381 = {const_25, node_380};
    Call node_381(Op::Get("add"), args_node_381, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Reshape_1 (Reshape)
    std::vector<Expr> args_node_382 = {node_381, node_371};
    Call node_382(Op::Get("reshape"), args_node_382, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.2/attention/v_lin/MatMul (MatMul)
    std::vector<Expr> args_node_383 = {node_354, const_79};
    Call node_383(Op::Get("matmul"), args_node_383, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/v_lin/Add (Add)
    std::vector<Expr> args_node_384 = {const_26, node_383};
    Call node_384(Op::Get("add"), args_node_384, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Reshape_2 (Reshape)
    std::vector<Expr> args_node_385 = {node_384, node_377};
    Call node_385(Op::Get("reshape"), args_node_385, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.2/attention/Transpose_1 (Transpose)
    std::vector<Expr> args_node_386 = {node_385};
    Call node_386(Op::Get("transpose"), args_node_386, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: /distilbert/transformer/layer.2/attention/Shape_1 (Shape)
    std::vector<Expr> args_node_387 = {node_379};
    Call node_387(Op::Get("shape"), args_node_387, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Constant_10 (Constant)
    Tensor node_388_tensor({1}, "int64");
    Constant node_388(node_388_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Constant_11 (Constant)
    Tensor node_389_tensor({1}, "int64");
    Constant node_389(node_389_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Slice (Slice)
    std::vector<Expr> args_node_390 = {node_387, node_388, node_389};
    Call node_390(Op::Get("slice"), args_node_390, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Cast (Cast)
    std::vector<Expr> args_node_391 = {node_390};
    Call node_391(Op::Get("cast"), args_node_391, CastAttrs::Create(1));
    // Node: /distilbert/transformer/layer.2/attention/Sqrt (Sqrt)
    std::vector<Expr> args_node_392 = {node_391};
    Call node_392(Op::Get("sqrt"), args_node_392, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Constant_12 (Constant)
    Tensor node_393_tensor({1}, "float32");
    Constant node_393(node_393_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Div (Div)
    std::vector<Expr> args_node_394 = {node_393, node_392};
    Call node_394(Op::Get("divide"), args_node_394, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.2/attention/Cast_1 (Cast)
    std::vector<Expr> args_node_395 = {node_394};
    Call node_395(Op::Get("cast"), args_node_395, CastAttrs::Create(1));
    // Node: /distilbert/transformer/layer.2/attention/Transpose_2 (Transpose)
    std::vector<Expr> args_node_396 = {node_382};
    Call node_396(Op::Get("transpose"), args_node_396, TransposeAttrs::Create({0, 2, 3, 1}));
    // Node: /distilbert/transformer/layer.2/attention/Sqrt_1 (Sqrt)
    std::vector<Expr> args_node_397 = {node_395};
    Call node_397(Op::Get("sqrt"), args_node_397, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Mul (Mul)
    std::vector<Expr> args_node_398 = {node_379, node_397};
    Call node_398(Op::Get("mul"), args_node_398, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Sqrt_2 (Sqrt)
    std::vector<Expr> args_node_399 = {node_395};
    Call node_399(Op::Get("sqrt"), args_node_399, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Mul_1 (Mul)
    std::vector<Expr> args_node_400 = {node_396, node_399};
    Call node_400(Op::Get("mul"), args_node_400, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/MatMul (MatMul)
    std::vector<Expr> args_node_401 = {node_398, node_400};
    Call node_401(Op::Get("matmul"), args_node_401, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Add (Add)
    std::vector<Expr> args_node_402 = {node_401, node_164};
    Call node_402(Op::Get("add"), args_node_402, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Softmax (Softmax)
    std::vector<Expr> args_node_403 = {node_402};
    Call node_403(Op::Get("softmax"), args_node_403, SoftmaxAttrs::Create(-1));
    // Node: /distilbert/transformer/layer.2/attention/MatMul_1 (MatMul)
    std::vector<Expr> args_node_404 = {node_403, node_386};
    Call node_404(Op::Get("matmul"), args_node_404, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Transpose_3 (Transpose)
    std::vector<Expr> args_node_405 = {node_404};
    Call node_405(Op::Get("transpose"), args_node_405, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: Constant_482 (Constant)
    Tensor node_406_tensor({1}, "int64");
    Constant node_406(node_406_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Unsqueeze_3 (Unsqueeze)
    std::vector<Expr> args_node_407 = {node_357, node_406};
    Call node_407(Op::Get("unsqueeze"), args_node_407, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/Constant_13 (Constant)
    Tensor node_408_tensor({1}, "int64");
    Constant node_408(node_408_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Constant_14 (Constant)
    Tensor node_409_tensor({1}, "int64");
    Constant node_409(node_409_tensor);
    // Node: /distilbert/transformer/layer.2/attention/Concat_3 (Concat)
    std::vector<Expr> args_node_410 = {node_407, node_408, node_409};
    Call node_410(Op::Get("concatenate"), args_node_410, ConcatAttrs::Create(0));
    // Node: /distilbert/transformer/layer.2/attention/Reshape_3 (Reshape)
    std::vector<Expr> args_node_411 = {node_405, node_410};
    Call node_411(Op::Get("reshape"), args_node_411, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.2/attention/out_lin/MatMul (MatMul)
    std::vector<Expr> args_node_412 = {node_411, const_80};
    Call node_412(Op::Get("matmul"), args_node_412, ObjectRef());
    // Node: /distilbert/transformer/layer.2/attention/out_lin/Add (Add)
    std::vector<Expr> args_node_413 = {const_27, node_412};
    Call node_413(Op::Get("add"), args_node_413, ObjectRef());
    // Node: /distilbert/transformer/layer.2/Add (Add)
    std::vector<Expr> args_node_414 = {node_413, node_354};
    Call node_414(Op::Get("add"), args_node_414, ObjectRef());
    // Node: /distilbert/transformer/layer.2/sa_layer_norm/ReduceMean (ReduceMean)
    std::vector<Expr> args_node_415 = {node_414};
    Call node_415(Op::Get("reduce_mean"), args_node_415, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.2/sa_layer_norm/Sub (Sub)
    std::vector<Expr> args_node_416 = {node_414, node_415};
    Call node_416(Op::Get("sub"), args_node_416, ObjectRef());
    // Node: /distilbert/transformer/layer.2/sa_layer_norm/Constant (Constant)
    Tensor node_417_tensor({}, "float32");
    Constant node_417(node_417_tensor);
    // Node: /distilbert/transformer/layer.2/sa_layer_norm/Pow (Pow)
    std::vector<Expr> args_node_418 = {node_416, node_417};
    Call node_418(Op::Get("pow"), args_node_418, ObjectRef());
    // Node: /distilbert/transformer/layer.2/sa_layer_norm/ReduceMean_1 (ReduceMean)
    std::vector<Expr> args_node_419 = {node_418};
    Call node_419(Op::Get("reduce_mean"), args_node_419, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.2/sa_layer_norm/Constant_1 (Constant)
    Tensor node_420_tensor({}, "float32");
    Constant node_420(node_420_tensor);
    // Node: /distilbert/transformer/layer.2/sa_layer_norm/Add (Add)
    std::vector<Expr> args_node_421 = {node_419, node_420};
    Call node_421(Op::Get("add"), args_node_421, ObjectRef());
    // Node: /distilbert/transformer/layer.2/sa_layer_norm/Sqrt (Sqrt)
    std::vector<Expr> args_node_422 = {node_421};
    Call node_422(Op::Get("sqrt"), args_node_422, ObjectRef());
    // Node: /distilbert/transformer/layer.2/sa_layer_norm/Div (Div)
    std::vector<Expr> args_node_423 = {node_416, node_422};
    Call node_423(Op::Get("divide"), args_node_423, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.2/sa_layer_norm/Mul (Mul)
    std::vector<Expr> args_node_424 = {node_423, const_28};
    Call node_424(Op::Get("mul"), args_node_424, ObjectRef());
    // Node: /distilbert/transformer/layer.2/sa_layer_norm/Add_1 (Add)
    std::vector<Expr> args_node_425 = {node_424, const_29};
    Call node_425(Op::Get("add"), args_node_425, ObjectRef());
    // Node: /distilbert/transformer/layer.2/ffn/lin1/MatMul (MatMul)
    std::vector<Expr> args_node_426 = {node_425, const_81};
    Call node_426(Op::Get("matmul"), args_node_426, ObjectRef());
    // Node: /distilbert/transformer/layer.2/ffn/lin1/Add (Add)
    std::vector<Expr> args_node_427 = {const_30, node_426};
    Call node_427(Op::Get("add"), args_node_427, ObjectRef());
    // Node: /distilbert/transformer/layer.2/ffn/activation/Constant (Constant)
    Tensor node_428_tensor({}, "float32");
    Constant node_428(node_428_tensor);
    // Node: /distilbert/transformer/layer.2/ffn/activation/Div (Div)
    std::vector<Expr> args_node_429 = {node_427, node_428};
    Call node_429(Op::Get("divide"), args_node_429, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.2/ffn/activation/Erf (Erf)
    std::vector<Expr> args_node_430 = {node_429};
    Call node_430(Op::Get("erf"), args_node_430, ErfAttrs::Create());
    // Node: /distilbert/transformer/layer.2/ffn/activation/Constant_1 (Constant)
    Tensor node_431_tensor({}, "float32");
    Constant node_431(node_431_tensor);
    // Node: /distilbert/transformer/layer.2/ffn/activation/Add (Add)
    std::vector<Expr> args_node_432 = {node_430, node_431};
    Call node_432(Op::Get("add"), args_node_432, ObjectRef());
    // Node: /distilbert/transformer/layer.2/ffn/activation/Mul (Mul)
    std::vector<Expr> args_node_433 = {node_427, node_432};
    Call node_433(Op::Get("mul"), args_node_433, ObjectRef());
    // Node: /distilbert/transformer/layer.2/ffn/activation/Constant_2 (Constant)
    Tensor node_434_tensor({}, "float32");
    Constant node_434(node_434_tensor);
    // Node: /distilbert/transformer/layer.2/ffn/activation/Mul_1 (Mul)
    std::vector<Expr> args_node_435 = {node_433, node_434};
    Call node_435(Op::Get("mul"), args_node_435, ObjectRef());
    // Node: /distilbert/transformer/layer.2/ffn/lin2/MatMul (MatMul)
    std::vector<Expr> args_node_436 = {node_435, const_82};
    Call node_436(Op::Get("matmul"), args_node_436, ObjectRef());
    // Node: /distilbert/transformer/layer.2/ffn/lin2/Add (Add)
    std::vector<Expr> args_node_437 = {const_31, node_436};
    Call node_437(Op::Get("add"), args_node_437, ObjectRef());
    // Node: /distilbert/transformer/layer.2/Add_1 (Add)
    std::vector<Expr> args_node_438 = {node_437, node_425};
    Call node_438(Op::Get("add"), args_node_438, ObjectRef());
    // Node: /distilbert/transformer/layer.2/output_layer_norm/ReduceMean (ReduceMean)
    std::vector<Expr> args_node_439 = {node_438};
    Call node_439(Op::Get("reduce_mean"), args_node_439, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.2/output_layer_norm/Sub (Sub)
    std::vector<Expr> args_node_440 = {node_438, node_439};
    Call node_440(Op::Get("sub"), args_node_440, ObjectRef());
    // Node: /distilbert/transformer/layer.2/output_layer_norm/Constant (Constant)
    Tensor node_441_tensor({}, "float32");
    Constant node_441(node_441_tensor);
    // Node: /distilbert/transformer/layer.2/output_layer_norm/Pow (Pow)
    std::vector<Expr> args_node_442 = {node_440, node_441};
    Call node_442(Op::Get("pow"), args_node_442, ObjectRef());
    // Node: /distilbert/transformer/layer.2/output_layer_norm/ReduceMean_1 (ReduceMean)
    std::vector<Expr> args_node_443 = {node_442};
    Call node_443(Op::Get("reduce_mean"), args_node_443, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.2/output_layer_norm/Constant_1 (Constant)
    Tensor node_444_tensor({}, "float32");
    Constant node_444(node_444_tensor);
    // Node: /distilbert/transformer/layer.2/output_layer_norm/Add (Add)
    std::vector<Expr> args_node_445 = {node_443, node_444};
    Call node_445(Op::Get("add"), args_node_445, ObjectRef());
    // Node: /distilbert/transformer/layer.2/output_layer_norm/Sqrt (Sqrt)
    std::vector<Expr> args_node_446 = {node_445};
    Call node_446(Op::Get("sqrt"), args_node_446, ObjectRef());
    // Node: /distilbert/transformer/layer.2/output_layer_norm/Div (Div)
    std::vector<Expr> args_node_447 = {node_440, node_446};
    Call node_447(Op::Get("divide"), args_node_447, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.2/output_layer_norm/Mul (Mul)
    std::vector<Expr> args_node_448 = {node_447, const_32};
    Call node_448(Op::Get("mul"), args_node_448, ObjectRef());
    // Node: /distilbert/transformer/layer.2/output_layer_norm/Add_1 (Add)
    std::vector<Expr> args_node_449 = {node_448, const_33};
    Call node_449(Op::Get("add"), args_node_449, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Shape (Shape)
    std::vector<Expr> args_node_450 = {node_449};
    Call node_450(Op::Get("shape"), args_node_450, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Constant (Constant)
    Tensor node_451_tensor({}, "int64");
    Constant node_451(node_451_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Gather (Gather)
    std::vector<Expr> args_node_452 = {node_450, node_451};
    Call node_452(Op::Get("gather"), args_node_452, GatherAttrs::Create(0));
    // Node: /distilbert/transformer/layer.3/attention/q_lin/MatMul (MatMul)
    std::vector<Expr> args_node_453 = {node_449, const_83};
    Call node_453(Op::Get("matmul"), args_node_453, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/q_lin/Add (Add)
    std::vector<Expr> args_node_454 = {const_34, node_453};
    Call node_454(Op::Get("add"), args_node_454, ObjectRef());
    // Node: Constant_531 (Constant)
    Tensor node_455_tensor({1}, "int64");
    Constant node_455(node_455_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Unsqueeze (Unsqueeze)
    std::vector<Expr> args_node_456 = {node_452, node_455};
    Call node_456(Op::Get("unsqueeze"), args_node_456, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Constant_1 (Constant)
    Tensor node_457_tensor({1}, "int64");
    Constant node_457(node_457_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Constant_2 (Constant)
    Tensor node_458_tensor({1}, "int64");
    Constant node_458(node_458_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Constant_3 (Constant)
    Tensor node_459_tensor({1}, "int64");
    Constant node_459(node_459_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Concat (Concat)
    std::vector<Expr> args_node_460 = {node_456, node_457, node_458, node_459};
    Call node_460(Op::Get("concatenate"), args_node_460, ConcatAttrs::Create(0));
    // Node: Constant_537 (Constant)
    Tensor node_461_tensor({1}, "int64");
    Constant node_461(node_461_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Unsqueeze_1 (Unsqueeze)
    std::vector<Expr> args_node_462 = {node_452, node_461};
    Call node_462(Op::Get("unsqueeze"), args_node_462, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Constant_4 (Constant)
    Tensor node_463_tensor({1}, "int64");
    Constant node_463(node_463_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Constant_5 (Constant)
    Tensor node_464_tensor({1}, "int64");
    Constant node_464(node_464_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Constant_6 (Constant)
    Tensor node_465_tensor({1}, "int64");
    Constant node_465(node_465_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Concat_1 (Concat)
    std::vector<Expr> args_node_466 = {node_462, node_463, node_464, node_465};
    Call node_466(Op::Get("concatenate"), args_node_466, ConcatAttrs::Create(0));
    // Node: Constant_543 (Constant)
    Tensor node_467_tensor({1}, "int64");
    Constant node_467(node_467_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Unsqueeze_2 (Unsqueeze)
    std::vector<Expr> args_node_468 = {node_452, node_467};
    Call node_468(Op::Get("unsqueeze"), args_node_468, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Constant_7 (Constant)
    Tensor node_469_tensor({1}, "int64");
    Constant node_469(node_469_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Constant_8 (Constant)
    Tensor node_470_tensor({1}, "int64");
    Constant node_470(node_470_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Constant_9 (Constant)
    Tensor node_471_tensor({1}, "int64");
    Constant node_471(node_471_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Concat_2 (Concat)
    std::vector<Expr> args_node_472 = {node_468, node_469, node_470, node_471};
    Call node_472(Op::Get("concatenate"), args_node_472, ConcatAttrs::Create(0));
    // Node: /distilbert/transformer/layer.3/attention/Reshape (Reshape)
    std::vector<Expr> args_node_473 = {node_454, node_460};
    Call node_473(Op::Get("reshape"), args_node_473, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.3/attention/Transpose (Transpose)
    std::vector<Expr> args_node_474 = {node_473};
    Call node_474(Op::Get("transpose"), args_node_474, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: /distilbert/transformer/layer.3/attention/k_lin/MatMul (MatMul)
    std::vector<Expr> args_node_475 = {node_449, const_84};
    Call node_475(Op::Get("matmul"), args_node_475, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/k_lin/Add (Add)
    std::vector<Expr> args_node_476 = {const_35, node_475};
    Call node_476(Op::Get("add"), args_node_476, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Reshape_1 (Reshape)
    std::vector<Expr> args_node_477 = {node_476, node_466};
    Call node_477(Op::Get("reshape"), args_node_477, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.3/attention/v_lin/MatMul (MatMul)
    std::vector<Expr> args_node_478 = {node_449, const_85};
    Call node_478(Op::Get("matmul"), args_node_478, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/v_lin/Add (Add)
    std::vector<Expr> args_node_479 = {const_36, node_478};
    Call node_479(Op::Get("add"), args_node_479, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Reshape_2 (Reshape)
    std::vector<Expr> args_node_480 = {node_479, node_472};
    Call node_480(Op::Get("reshape"), args_node_480, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.3/attention/Transpose_1 (Transpose)
    std::vector<Expr> args_node_481 = {node_480};
    Call node_481(Op::Get("transpose"), args_node_481, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: /distilbert/transformer/layer.3/attention/Shape_1 (Shape)
    std::vector<Expr> args_node_482 = {node_474};
    Call node_482(Op::Get("shape"), args_node_482, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Constant_10 (Constant)
    Tensor node_483_tensor({1}, "int64");
    Constant node_483(node_483_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Constant_11 (Constant)
    Tensor node_484_tensor({1}, "int64");
    Constant node_484(node_484_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Slice (Slice)
    std::vector<Expr> args_node_485 = {node_482, node_483, node_484};
    Call node_485(Op::Get("slice"), args_node_485, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Cast (Cast)
    std::vector<Expr> args_node_486 = {node_485};
    Call node_486(Op::Get("cast"), args_node_486, CastAttrs::Create(1));
    // Node: /distilbert/transformer/layer.3/attention/Sqrt (Sqrt)
    std::vector<Expr> args_node_487 = {node_486};
    Call node_487(Op::Get("sqrt"), args_node_487, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Constant_12 (Constant)
    Tensor node_488_tensor({1}, "float32");
    Constant node_488(node_488_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Div (Div)
    std::vector<Expr> args_node_489 = {node_488, node_487};
    Call node_489(Op::Get("divide"), args_node_489, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.3/attention/Cast_1 (Cast)
    std::vector<Expr> args_node_490 = {node_489};
    Call node_490(Op::Get("cast"), args_node_490, CastAttrs::Create(1));
    // Node: /distilbert/transformer/layer.3/attention/Transpose_2 (Transpose)
    std::vector<Expr> args_node_491 = {node_477};
    Call node_491(Op::Get("transpose"), args_node_491, TransposeAttrs::Create({0, 2, 3, 1}));
    // Node: /distilbert/transformer/layer.3/attention/Sqrt_1 (Sqrt)
    std::vector<Expr> args_node_492 = {node_490};
    Call node_492(Op::Get("sqrt"), args_node_492, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Mul (Mul)
    std::vector<Expr> args_node_493 = {node_474, node_492};
    Call node_493(Op::Get("mul"), args_node_493, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Sqrt_2 (Sqrt)
    std::vector<Expr> args_node_494 = {node_490};
    Call node_494(Op::Get("sqrt"), args_node_494, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Mul_1 (Mul)
    std::vector<Expr> args_node_495 = {node_491, node_494};
    Call node_495(Op::Get("mul"), args_node_495, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/MatMul (MatMul)
    std::vector<Expr> args_node_496 = {node_493, node_495};
    Call node_496(Op::Get("matmul"), args_node_496, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Add (Add)
    std::vector<Expr> args_node_497 = {node_496, node_164};
    Call node_497(Op::Get("add"), args_node_497, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Softmax (Softmax)
    std::vector<Expr> args_node_498 = {node_497};
    Call node_498(Op::Get("softmax"), args_node_498, SoftmaxAttrs::Create(-1));
    // Node: /distilbert/transformer/layer.3/attention/MatMul_1 (MatMul)
    std::vector<Expr> args_node_499 = {node_498, node_481};
    Call node_499(Op::Get("matmul"), args_node_499, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Transpose_3 (Transpose)
    std::vector<Expr> args_node_500 = {node_499};
    Call node_500(Op::Get("transpose"), args_node_500, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: Constant_577 (Constant)
    Tensor node_501_tensor({1}, "int64");
    Constant node_501(node_501_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Unsqueeze_3 (Unsqueeze)
    std::vector<Expr> args_node_502 = {node_452, node_501};
    Call node_502(Op::Get("unsqueeze"), args_node_502, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/Constant_13 (Constant)
    Tensor node_503_tensor({1}, "int64");
    Constant node_503(node_503_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Constant_14 (Constant)
    Tensor node_504_tensor({1}, "int64");
    Constant node_504(node_504_tensor);
    // Node: /distilbert/transformer/layer.3/attention/Concat_3 (Concat)
    std::vector<Expr> args_node_505 = {node_502, node_503, node_504};
    Call node_505(Op::Get("concatenate"), args_node_505, ConcatAttrs::Create(0));
    // Node: /distilbert/transformer/layer.3/attention/Reshape_3 (Reshape)
    std::vector<Expr> args_node_506 = {node_500, node_505};
    Call node_506(Op::Get("reshape"), args_node_506, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.3/attention/out_lin/MatMul (MatMul)
    std::vector<Expr> args_node_507 = {node_506, const_86};
    Call node_507(Op::Get("matmul"), args_node_507, ObjectRef());
    // Node: /distilbert/transformer/layer.3/attention/out_lin/Add (Add)
    std::vector<Expr> args_node_508 = {const_37, node_507};
    Call node_508(Op::Get("add"), args_node_508, ObjectRef());
    // Node: /distilbert/transformer/layer.3/Add (Add)
    std::vector<Expr> args_node_509 = {node_508, node_449};
    Call node_509(Op::Get("add"), args_node_509, ObjectRef());
    // Node: /distilbert/transformer/layer.3/sa_layer_norm/ReduceMean (ReduceMean)
    std::vector<Expr> args_node_510 = {node_509};
    Call node_510(Op::Get("reduce_mean"), args_node_510, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.3/sa_layer_norm/Sub (Sub)
    std::vector<Expr> args_node_511 = {node_509, node_510};
    Call node_511(Op::Get("sub"), args_node_511, ObjectRef());
    // Node: /distilbert/transformer/layer.3/sa_layer_norm/Constant (Constant)
    Tensor node_512_tensor({}, "float32");
    Constant node_512(node_512_tensor);
    // Node: /distilbert/transformer/layer.3/sa_layer_norm/Pow (Pow)
    std::vector<Expr> args_node_513 = {node_511, node_512};
    Call node_513(Op::Get("pow"), args_node_513, ObjectRef());
    // Node: /distilbert/transformer/layer.3/sa_layer_norm/ReduceMean_1 (ReduceMean)
    std::vector<Expr> args_node_514 = {node_513};
    Call node_514(Op::Get("reduce_mean"), args_node_514, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.3/sa_layer_norm/Constant_1 (Constant)
    Tensor node_515_tensor({}, "float32");
    Constant node_515(node_515_tensor);
    // Node: /distilbert/transformer/layer.3/sa_layer_norm/Add (Add)
    std::vector<Expr> args_node_516 = {node_514, node_515};
    Call node_516(Op::Get("add"), args_node_516, ObjectRef());
    // Node: /distilbert/transformer/layer.3/sa_layer_norm/Sqrt (Sqrt)
    std::vector<Expr> args_node_517 = {node_516};
    Call node_517(Op::Get("sqrt"), args_node_517, ObjectRef());
    // Node: /distilbert/transformer/layer.3/sa_layer_norm/Div (Div)
    std::vector<Expr> args_node_518 = {node_511, node_517};
    Call node_518(Op::Get("divide"), args_node_518, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.3/sa_layer_norm/Mul (Mul)
    std::vector<Expr> args_node_519 = {node_518, const_38};
    Call node_519(Op::Get("mul"), args_node_519, ObjectRef());
    // Node: /distilbert/transformer/layer.3/sa_layer_norm/Add_1 (Add)
    std::vector<Expr> args_node_520 = {node_519, const_39};
    Call node_520(Op::Get("add"), args_node_520, ObjectRef());
    // Node: /distilbert/transformer/layer.3/ffn/lin1/MatMul (MatMul)
    std::vector<Expr> args_node_521 = {node_520, const_87};
    Call node_521(Op::Get("matmul"), args_node_521, ObjectRef());
    // Node: /distilbert/transformer/layer.3/ffn/lin1/Add (Add)
    std::vector<Expr> args_node_522 = {const_40, node_521};
    Call node_522(Op::Get("add"), args_node_522, ObjectRef());
    // Node: /distilbert/transformer/layer.3/ffn/activation/Constant (Constant)
    Tensor node_523_tensor({}, "float32");
    Constant node_523(node_523_tensor);
    // Node: /distilbert/transformer/layer.3/ffn/activation/Div (Div)
    std::vector<Expr> args_node_524 = {node_522, node_523};
    Call node_524(Op::Get("divide"), args_node_524, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.3/ffn/activation/Erf (Erf)
    std::vector<Expr> args_node_525 = {node_524};
    Call node_525(Op::Get("erf"), args_node_525, ErfAttrs::Create());
    // Node: /distilbert/transformer/layer.3/ffn/activation/Constant_1 (Constant)
    Tensor node_526_tensor({}, "float32");
    Constant node_526(node_526_tensor);
    // Node: /distilbert/transformer/layer.3/ffn/activation/Add (Add)
    std::vector<Expr> args_node_527 = {node_525, node_526};
    Call node_527(Op::Get("add"), args_node_527, ObjectRef());
    // Node: /distilbert/transformer/layer.3/ffn/activation/Mul (Mul)
    std::vector<Expr> args_node_528 = {node_522, node_527};
    Call node_528(Op::Get("mul"), args_node_528, ObjectRef());
    // Node: /distilbert/transformer/layer.3/ffn/activation/Constant_2 (Constant)
    Tensor node_529_tensor({}, "float32");
    Constant node_529(node_529_tensor);
    // Node: /distilbert/transformer/layer.3/ffn/activation/Mul_1 (Mul)
    std::vector<Expr> args_node_530 = {node_528, node_529};
    Call node_530(Op::Get("mul"), args_node_530, ObjectRef());
    // Node: /distilbert/transformer/layer.3/ffn/lin2/MatMul (MatMul)
    std::vector<Expr> args_node_531 = {node_530, const_88};
    Call node_531(Op::Get("matmul"), args_node_531, ObjectRef());
    // Node: /distilbert/transformer/layer.3/ffn/lin2/Add (Add)
    std::vector<Expr> args_node_532 = {const_41, node_531};
    Call node_532(Op::Get("add"), args_node_532, ObjectRef());
    // Node: /distilbert/transformer/layer.3/Add_1 (Add)
    std::vector<Expr> args_node_533 = {node_532, node_520};
    Call node_533(Op::Get("add"), args_node_533, ObjectRef());
    // Node: /distilbert/transformer/layer.3/output_layer_norm/ReduceMean (ReduceMean)
    std::vector<Expr> args_node_534 = {node_533};
    Call node_534(Op::Get("reduce_mean"), args_node_534, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.3/output_layer_norm/Sub (Sub)
    std::vector<Expr> args_node_535 = {node_533, node_534};
    Call node_535(Op::Get("sub"), args_node_535, ObjectRef());
    // Node: /distilbert/transformer/layer.3/output_layer_norm/Constant (Constant)
    Tensor node_536_tensor({}, "float32");
    Constant node_536(node_536_tensor);
    // Node: /distilbert/transformer/layer.3/output_layer_norm/Pow (Pow)
    std::vector<Expr> args_node_537 = {node_535, node_536};
    Call node_537(Op::Get("pow"), args_node_537, ObjectRef());
    // Node: /distilbert/transformer/layer.3/output_layer_norm/ReduceMean_1 (ReduceMean)
    std::vector<Expr> args_node_538 = {node_537};
    Call node_538(Op::Get("reduce_mean"), args_node_538, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.3/output_layer_norm/Constant_1 (Constant)
    Tensor node_539_tensor({}, "float32");
    Constant node_539(node_539_tensor);
    // Node: /distilbert/transformer/layer.3/output_layer_norm/Add (Add)
    std::vector<Expr> args_node_540 = {node_538, node_539};
    Call node_540(Op::Get("add"), args_node_540, ObjectRef());
    // Node: /distilbert/transformer/layer.3/output_layer_norm/Sqrt (Sqrt)
    std::vector<Expr> args_node_541 = {node_540};
    Call node_541(Op::Get("sqrt"), args_node_541, ObjectRef());
    // Node: /distilbert/transformer/layer.3/output_layer_norm/Div (Div)
    std::vector<Expr> args_node_542 = {node_535, node_541};
    Call node_542(Op::Get("divide"), args_node_542, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.3/output_layer_norm/Mul (Mul)
    std::vector<Expr> args_node_543 = {node_542, const_42};
    Call node_543(Op::Get("mul"), args_node_543, ObjectRef());
    // Node: /distilbert/transformer/layer.3/output_layer_norm/Add_1 (Add)
    std::vector<Expr> args_node_544 = {node_543, const_43};
    Call node_544(Op::Get("add"), args_node_544, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Shape (Shape)
    std::vector<Expr> args_node_545 = {node_544};
    Call node_545(Op::Get("shape"), args_node_545, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Constant (Constant)
    Tensor node_546_tensor({}, "int64");
    Constant node_546(node_546_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Gather (Gather)
    std::vector<Expr> args_node_547 = {node_545, node_546};
    Call node_547(Op::Get("gather"), args_node_547, GatherAttrs::Create(0));
    // Node: /distilbert/transformer/layer.4/attention/q_lin/MatMul (MatMul)
    std::vector<Expr> args_node_548 = {node_544, const_89};
    Call node_548(Op::Get("matmul"), args_node_548, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/q_lin/Add (Add)
    std::vector<Expr> args_node_549 = {const_44, node_548};
    Call node_549(Op::Get("add"), args_node_549, ObjectRef());
    // Node: Constant_626 (Constant)
    Tensor node_550_tensor({1}, "int64");
    Constant node_550(node_550_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Unsqueeze (Unsqueeze)
    std::vector<Expr> args_node_551 = {node_547, node_550};
    Call node_551(Op::Get("unsqueeze"), args_node_551, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Constant_1 (Constant)
    Tensor node_552_tensor({1}, "int64");
    Constant node_552(node_552_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Constant_2 (Constant)
    Tensor node_553_tensor({1}, "int64");
    Constant node_553(node_553_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Constant_3 (Constant)
    Tensor node_554_tensor({1}, "int64");
    Constant node_554(node_554_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Concat (Concat)
    std::vector<Expr> args_node_555 = {node_551, node_552, node_553, node_554};
    Call node_555(Op::Get("concatenate"), args_node_555, ConcatAttrs::Create(0));
    // Node: Constant_632 (Constant)
    Tensor node_556_tensor({1}, "int64");
    Constant node_556(node_556_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Unsqueeze_1 (Unsqueeze)
    std::vector<Expr> args_node_557 = {node_547, node_556};
    Call node_557(Op::Get("unsqueeze"), args_node_557, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Constant_4 (Constant)
    Tensor node_558_tensor({1}, "int64");
    Constant node_558(node_558_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Constant_5 (Constant)
    Tensor node_559_tensor({1}, "int64");
    Constant node_559(node_559_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Constant_6 (Constant)
    Tensor node_560_tensor({1}, "int64");
    Constant node_560(node_560_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Concat_1 (Concat)
    std::vector<Expr> args_node_561 = {node_557, node_558, node_559, node_560};
    Call node_561(Op::Get("concatenate"), args_node_561, ConcatAttrs::Create(0));
    // Node: Constant_638 (Constant)
    Tensor node_562_tensor({1}, "int64");
    Constant node_562(node_562_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Unsqueeze_2 (Unsqueeze)
    std::vector<Expr> args_node_563 = {node_547, node_562};
    Call node_563(Op::Get("unsqueeze"), args_node_563, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Constant_7 (Constant)
    Tensor node_564_tensor({1}, "int64");
    Constant node_564(node_564_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Constant_8 (Constant)
    Tensor node_565_tensor({1}, "int64");
    Constant node_565(node_565_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Constant_9 (Constant)
    Tensor node_566_tensor({1}, "int64");
    Constant node_566(node_566_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Concat_2 (Concat)
    std::vector<Expr> args_node_567 = {node_563, node_564, node_565, node_566};
    Call node_567(Op::Get("concatenate"), args_node_567, ConcatAttrs::Create(0));
    // Node: /distilbert/transformer/layer.4/attention/Reshape (Reshape)
    std::vector<Expr> args_node_568 = {node_549, node_555};
    Call node_568(Op::Get("reshape"), args_node_568, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.4/attention/Transpose (Transpose)
    std::vector<Expr> args_node_569 = {node_568};
    Call node_569(Op::Get("transpose"), args_node_569, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: /distilbert/transformer/layer.4/attention/k_lin/MatMul (MatMul)
    std::vector<Expr> args_node_570 = {node_544, const_90};
    Call node_570(Op::Get("matmul"), args_node_570, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/k_lin/Add (Add)
    std::vector<Expr> args_node_571 = {const_45, node_570};
    Call node_571(Op::Get("add"), args_node_571, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Reshape_1 (Reshape)
    std::vector<Expr> args_node_572 = {node_571, node_561};
    Call node_572(Op::Get("reshape"), args_node_572, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.4/attention/v_lin/MatMul (MatMul)
    std::vector<Expr> args_node_573 = {node_544, const_91};
    Call node_573(Op::Get("matmul"), args_node_573, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/v_lin/Add (Add)
    std::vector<Expr> args_node_574 = {const_46, node_573};
    Call node_574(Op::Get("add"), args_node_574, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Reshape_2 (Reshape)
    std::vector<Expr> args_node_575 = {node_574, node_567};
    Call node_575(Op::Get("reshape"), args_node_575, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.4/attention/Transpose_1 (Transpose)
    std::vector<Expr> args_node_576 = {node_575};
    Call node_576(Op::Get("transpose"), args_node_576, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: /distilbert/transformer/layer.4/attention/Shape_1 (Shape)
    std::vector<Expr> args_node_577 = {node_569};
    Call node_577(Op::Get("shape"), args_node_577, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Constant_10 (Constant)
    Tensor node_578_tensor({1}, "int64");
    Constant node_578(node_578_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Constant_11 (Constant)
    Tensor node_579_tensor({1}, "int64");
    Constant node_579(node_579_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Slice (Slice)
    std::vector<Expr> args_node_580 = {node_577, node_578, node_579};
    Call node_580(Op::Get("slice"), args_node_580, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Cast (Cast)
    std::vector<Expr> args_node_581 = {node_580};
    Call node_581(Op::Get("cast"), args_node_581, CastAttrs::Create(1));
    // Node: /distilbert/transformer/layer.4/attention/Sqrt (Sqrt)
    std::vector<Expr> args_node_582 = {node_581};
    Call node_582(Op::Get("sqrt"), args_node_582, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Constant_12 (Constant)
    Tensor node_583_tensor({1}, "float32");
    Constant node_583(node_583_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Div (Div)
    std::vector<Expr> args_node_584 = {node_583, node_582};
    Call node_584(Op::Get("divide"), args_node_584, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.4/attention/Cast_1 (Cast)
    std::vector<Expr> args_node_585 = {node_584};
    Call node_585(Op::Get("cast"), args_node_585, CastAttrs::Create(1));
    // Node: /distilbert/transformer/layer.4/attention/Transpose_2 (Transpose)
    std::vector<Expr> args_node_586 = {node_572};
    Call node_586(Op::Get("transpose"), args_node_586, TransposeAttrs::Create({0, 2, 3, 1}));
    // Node: /distilbert/transformer/layer.4/attention/Sqrt_1 (Sqrt)
    std::vector<Expr> args_node_587 = {node_585};
    Call node_587(Op::Get("sqrt"), args_node_587, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Mul (Mul)
    std::vector<Expr> args_node_588 = {node_569, node_587};
    Call node_588(Op::Get("mul"), args_node_588, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Sqrt_2 (Sqrt)
    std::vector<Expr> args_node_589 = {node_585};
    Call node_589(Op::Get("sqrt"), args_node_589, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Mul_1 (Mul)
    std::vector<Expr> args_node_590 = {node_586, node_589};
    Call node_590(Op::Get("mul"), args_node_590, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/MatMul (MatMul)
    std::vector<Expr> args_node_591 = {node_588, node_590};
    Call node_591(Op::Get("matmul"), args_node_591, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Add (Add)
    std::vector<Expr> args_node_592 = {node_591, node_164};
    Call node_592(Op::Get("add"), args_node_592, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Softmax (Softmax)
    std::vector<Expr> args_node_593 = {node_592};
    Call node_593(Op::Get("softmax"), args_node_593, SoftmaxAttrs::Create(-1));
    // Node: /distilbert/transformer/layer.4/attention/MatMul_1 (MatMul)
    std::vector<Expr> args_node_594 = {node_593, node_576};
    Call node_594(Op::Get("matmul"), args_node_594, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Transpose_3 (Transpose)
    std::vector<Expr> args_node_595 = {node_594};
    Call node_595(Op::Get("transpose"), args_node_595, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: Constant_672 (Constant)
    Tensor node_596_tensor({1}, "int64");
    Constant node_596(node_596_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Unsqueeze_3 (Unsqueeze)
    std::vector<Expr> args_node_597 = {node_547, node_596};
    Call node_597(Op::Get("unsqueeze"), args_node_597, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/Constant_13 (Constant)
    Tensor node_598_tensor({1}, "int64");
    Constant node_598(node_598_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Constant_14 (Constant)
    Tensor node_599_tensor({1}, "int64");
    Constant node_599(node_599_tensor);
    // Node: /distilbert/transformer/layer.4/attention/Concat_3 (Concat)
    std::vector<Expr> args_node_600 = {node_597, node_598, node_599};
    Call node_600(Op::Get("concatenate"), args_node_600, ConcatAttrs::Create(0));
    // Node: /distilbert/transformer/layer.4/attention/Reshape_3 (Reshape)
    std::vector<Expr> args_node_601 = {node_595, node_600};
    Call node_601(Op::Get("reshape"), args_node_601, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.4/attention/out_lin/MatMul (MatMul)
    std::vector<Expr> args_node_602 = {node_601, const_92};
    Call node_602(Op::Get("matmul"), args_node_602, ObjectRef());
    // Node: /distilbert/transformer/layer.4/attention/out_lin/Add (Add)
    std::vector<Expr> args_node_603 = {const_47, node_602};
    Call node_603(Op::Get("add"), args_node_603, ObjectRef());
    // Node: /distilbert/transformer/layer.4/Add (Add)
    std::vector<Expr> args_node_604 = {node_603, node_544};
    Call node_604(Op::Get("add"), args_node_604, ObjectRef());
    // Node: /distilbert/transformer/layer.4/sa_layer_norm/ReduceMean (ReduceMean)
    std::vector<Expr> args_node_605 = {node_604};
    Call node_605(Op::Get("reduce_mean"), args_node_605, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.4/sa_layer_norm/Sub (Sub)
    std::vector<Expr> args_node_606 = {node_604, node_605};
    Call node_606(Op::Get("sub"), args_node_606, ObjectRef());
    // Node: /distilbert/transformer/layer.4/sa_layer_norm/Constant (Constant)
    Tensor node_607_tensor({}, "float32");
    Constant node_607(node_607_tensor);
    // Node: /distilbert/transformer/layer.4/sa_layer_norm/Pow (Pow)
    std::vector<Expr> args_node_608 = {node_606, node_607};
    Call node_608(Op::Get("pow"), args_node_608, ObjectRef());
    // Node: /distilbert/transformer/layer.4/sa_layer_norm/ReduceMean_1 (ReduceMean)
    std::vector<Expr> args_node_609 = {node_608};
    Call node_609(Op::Get("reduce_mean"), args_node_609, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.4/sa_layer_norm/Constant_1 (Constant)
    Tensor node_610_tensor({}, "float32");
    Constant node_610(node_610_tensor);
    // Node: /distilbert/transformer/layer.4/sa_layer_norm/Add (Add)
    std::vector<Expr> args_node_611 = {node_609, node_610};
    Call node_611(Op::Get("add"), args_node_611, ObjectRef());
    // Node: /distilbert/transformer/layer.4/sa_layer_norm/Sqrt (Sqrt)
    std::vector<Expr> args_node_612 = {node_611};
    Call node_612(Op::Get("sqrt"), args_node_612, ObjectRef());
    // Node: /distilbert/transformer/layer.4/sa_layer_norm/Div (Div)
    std::vector<Expr> args_node_613 = {node_606, node_612};
    Call node_613(Op::Get("divide"), args_node_613, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.4/sa_layer_norm/Mul (Mul)
    std::vector<Expr> args_node_614 = {node_613, const_48};
    Call node_614(Op::Get("mul"), args_node_614, ObjectRef());
    // Node: /distilbert/transformer/layer.4/sa_layer_norm/Add_1 (Add)
    std::vector<Expr> args_node_615 = {node_614, const_49};
    Call node_615(Op::Get("add"), args_node_615, ObjectRef());
    // Node: /distilbert/transformer/layer.4/ffn/lin1/MatMul (MatMul)
    std::vector<Expr> args_node_616 = {node_615, const_93};
    Call node_616(Op::Get("matmul"), args_node_616, ObjectRef());
    // Node: /distilbert/transformer/layer.4/ffn/lin1/Add (Add)
    std::vector<Expr> args_node_617 = {const_50, node_616};
    Call node_617(Op::Get("add"), args_node_617, ObjectRef());
    // Node: /distilbert/transformer/layer.4/ffn/activation/Constant (Constant)
    Tensor node_618_tensor({}, "float32");
    Constant node_618(node_618_tensor);
    // Node: /distilbert/transformer/layer.4/ffn/activation/Div (Div)
    std::vector<Expr> args_node_619 = {node_617, node_618};
    Call node_619(Op::Get("divide"), args_node_619, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.4/ffn/activation/Erf (Erf)
    std::vector<Expr> args_node_620 = {node_619};
    Call node_620(Op::Get("erf"), args_node_620, ErfAttrs::Create());
    // Node: /distilbert/transformer/layer.4/ffn/activation/Constant_1 (Constant)
    Tensor node_621_tensor({}, "float32");
    Constant node_621(node_621_tensor);
    // Node: /distilbert/transformer/layer.4/ffn/activation/Add (Add)
    std::vector<Expr> args_node_622 = {node_620, node_621};
    Call node_622(Op::Get("add"), args_node_622, ObjectRef());
    // Node: /distilbert/transformer/layer.4/ffn/activation/Mul (Mul)
    std::vector<Expr> args_node_623 = {node_617, node_622};
    Call node_623(Op::Get("mul"), args_node_623, ObjectRef());
    // Node: /distilbert/transformer/layer.4/ffn/activation/Constant_2 (Constant)
    Tensor node_624_tensor({}, "float32");
    Constant node_624(node_624_tensor);
    // Node: /distilbert/transformer/layer.4/ffn/activation/Mul_1 (Mul)
    std::vector<Expr> args_node_625 = {node_623, node_624};
    Call node_625(Op::Get("mul"), args_node_625, ObjectRef());
    // Node: /distilbert/transformer/layer.4/ffn/lin2/MatMul (MatMul)
    std::vector<Expr> args_node_626 = {node_625, const_94};
    Call node_626(Op::Get("matmul"), args_node_626, ObjectRef());
    // Node: /distilbert/transformer/layer.4/ffn/lin2/Add (Add)
    std::vector<Expr> args_node_627 = {const_51, node_626};
    Call node_627(Op::Get("add"), args_node_627, ObjectRef());
    // Node: /distilbert/transformer/layer.4/Add_1 (Add)
    std::vector<Expr> args_node_628 = {node_627, node_615};
    Call node_628(Op::Get("add"), args_node_628, ObjectRef());
    // Node: /distilbert/transformer/layer.4/output_layer_norm/ReduceMean (ReduceMean)
    std::vector<Expr> args_node_629 = {node_628};
    Call node_629(Op::Get("reduce_mean"), args_node_629, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.4/output_layer_norm/Sub (Sub)
    std::vector<Expr> args_node_630 = {node_628, node_629};
    Call node_630(Op::Get("sub"), args_node_630, ObjectRef());
    // Node: /distilbert/transformer/layer.4/output_layer_norm/Constant (Constant)
    Tensor node_631_tensor({}, "float32");
    Constant node_631(node_631_tensor);
    // Node: /distilbert/transformer/layer.4/output_layer_norm/Pow (Pow)
    std::vector<Expr> args_node_632 = {node_630, node_631};
    Call node_632(Op::Get("pow"), args_node_632, ObjectRef());
    // Node: /distilbert/transformer/layer.4/output_layer_norm/ReduceMean_1 (ReduceMean)
    std::vector<Expr> args_node_633 = {node_632};
    Call node_633(Op::Get("reduce_mean"), args_node_633, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.4/output_layer_norm/Constant_1 (Constant)
    Tensor node_634_tensor({}, "float32");
    Constant node_634(node_634_tensor);
    // Node: /distilbert/transformer/layer.4/output_layer_norm/Add (Add)
    std::vector<Expr> args_node_635 = {node_633, node_634};
    Call node_635(Op::Get("add"), args_node_635, ObjectRef());
    // Node: /distilbert/transformer/layer.4/output_layer_norm/Sqrt (Sqrt)
    std::vector<Expr> args_node_636 = {node_635};
    Call node_636(Op::Get("sqrt"), args_node_636, ObjectRef());
    // Node: /distilbert/transformer/layer.4/output_layer_norm/Div (Div)
    std::vector<Expr> args_node_637 = {node_630, node_636};
    Call node_637(Op::Get("divide"), args_node_637, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.4/output_layer_norm/Mul (Mul)
    std::vector<Expr> args_node_638 = {node_637, const_52};
    Call node_638(Op::Get("mul"), args_node_638, ObjectRef());
    // Node: /distilbert/transformer/layer.4/output_layer_norm/Add_1 (Add)
    std::vector<Expr> args_node_639 = {node_638, const_53};
    Call node_639(Op::Get("add"), args_node_639, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Shape (Shape)
    std::vector<Expr> args_node_640 = {node_639};
    Call node_640(Op::Get("shape"), args_node_640, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Constant (Constant)
    Tensor node_641_tensor({}, "int64");
    Constant node_641(node_641_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Gather (Gather)
    std::vector<Expr> args_node_642 = {node_640, node_641};
    Call node_642(Op::Get("gather"), args_node_642, GatherAttrs::Create(0));
    // Node: /distilbert/transformer/layer.5/attention/q_lin/MatMul (MatMul)
    std::vector<Expr> args_node_643 = {node_639, const_95};
    Call node_643(Op::Get("matmul"), args_node_643, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/q_lin/Add (Add)
    std::vector<Expr> args_node_644 = {const_54, node_643};
    Call node_644(Op::Get("add"), args_node_644, ObjectRef());
    // Node: Constant_721 (Constant)
    Tensor node_645_tensor({1}, "int64");
    Constant node_645(node_645_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Unsqueeze (Unsqueeze)
    std::vector<Expr> args_node_646 = {node_642, node_645};
    Call node_646(Op::Get("unsqueeze"), args_node_646, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Constant_1 (Constant)
    Tensor node_647_tensor({1}, "int64");
    Constant node_647(node_647_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Constant_2 (Constant)
    Tensor node_648_tensor({1}, "int64");
    Constant node_648(node_648_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Constant_3 (Constant)
    Tensor node_649_tensor({1}, "int64");
    Constant node_649(node_649_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Concat (Concat)
    std::vector<Expr> args_node_650 = {node_646, node_647, node_648, node_649};
    Call node_650(Op::Get("concatenate"), args_node_650, ConcatAttrs::Create(0));
    // Node: Constant_727 (Constant)
    Tensor node_651_tensor({1}, "int64");
    Constant node_651(node_651_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Unsqueeze_1 (Unsqueeze)
    std::vector<Expr> args_node_652 = {node_642, node_651};
    Call node_652(Op::Get("unsqueeze"), args_node_652, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Constant_4 (Constant)
    Tensor node_653_tensor({1}, "int64");
    Constant node_653(node_653_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Constant_5 (Constant)
    Tensor node_654_tensor({1}, "int64");
    Constant node_654(node_654_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Constant_6 (Constant)
    Tensor node_655_tensor({1}, "int64");
    Constant node_655(node_655_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Concat_1 (Concat)
    std::vector<Expr> args_node_656 = {node_652, node_653, node_654, node_655};
    Call node_656(Op::Get("concatenate"), args_node_656, ConcatAttrs::Create(0));
    // Node: Constant_733 (Constant)
    Tensor node_657_tensor({1}, "int64");
    Constant node_657(node_657_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Unsqueeze_2 (Unsqueeze)
    std::vector<Expr> args_node_658 = {node_642, node_657};
    Call node_658(Op::Get("unsqueeze"), args_node_658, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Constant_7 (Constant)
    Tensor node_659_tensor({1}, "int64");
    Constant node_659(node_659_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Constant_8 (Constant)
    Tensor node_660_tensor({1}, "int64");
    Constant node_660(node_660_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Constant_9 (Constant)
    Tensor node_661_tensor({1}, "int64");
    Constant node_661(node_661_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Concat_2 (Concat)
    std::vector<Expr> args_node_662 = {node_658, node_659, node_660, node_661};
    Call node_662(Op::Get("concatenate"), args_node_662, ConcatAttrs::Create(0));
    // Node: /distilbert/transformer/layer.5/attention/Reshape (Reshape)
    std::vector<Expr> args_node_663 = {node_644, node_650};
    Call node_663(Op::Get("reshape"), args_node_663, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.5/attention/Transpose (Transpose)
    std::vector<Expr> args_node_664 = {node_663};
    Call node_664(Op::Get("transpose"), args_node_664, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: /distilbert/transformer/layer.5/attention/k_lin/MatMul (MatMul)
    std::vector<Expr> args_node_665 = {node_639, const_96};
    Call node_665(Op::Get("matmul"), args_node_665, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/k_lin/Add (Add)
    std::vector<Expr> args_node_666 = {const_55, node_665};
    Call node_666(Op::Get("add"), args_node_666, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Reshape_1 (Reshape)
    std::vector<Expr> args_node_667 = {node_666, node_656};
    Call node_667(Op::Get("reshape"), args_node_667, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.5/attention/v_lin/MatMul (MatMul)
    std::vector<Expr> args_node_668 = {node_639, const_97};
    Call node_668(Op::Get("matmul"), args_node_668, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/v_lin/Add (Add)
    std::vector<Expr> args_node_669 = {const_56, node_668};
    Call node_669(Op::Get("add"), args_node_669, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Reshape_2 (Reshape)
    std::vector<Expr> args_node_670 = {node_669, node_662};
    Call node_670(Op::Get("reshape"), args_node_670, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.5/attention/Transpose_1 (Transpose)
    std::vector<Expr> args_node_671 = {node_670};
    Call node_671(Op::Get("transpose"), args_node_671, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: /distilbert/transformer/layer.5/attention/Shape_1 (Shape)
    std::vector<Expr> args_node_672 = {node_664};
    Call node_672(Op::Get("shape"), args_node_672, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Constant_10 (Constant)
    Tensor node_673_tensor({1}, "int64");
    Constant node_673(node_673_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Constant_11 (Constant)
    Tensor node_674_tensor({1}, "int64");
    Constant node_674(node_674_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Slice (Slice)
    std::vector<Expr> args_node_675 = {node_672, node_673, node_674};
    Call node_675(Op::Get("slice"), args_node_675, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Cast (Cast)
    std::vector<Expr> args_node_676 = {node_675};
    Call node_676(Op::Get("cast"), args_node_676, CastAttrs::Create(1));
    // Node: /distilbert/transformer/layer.5/attention/Sqrt (Sqrt)
    std::vector<Expr> args_node_677 = {node_676};
    Call node_677(Op::Get("sqrt"), args_node_677, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Constant_12 (Constant)
    Tensor node_678_tensor({1}, "float32");
    Constant node_678(node_678_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Div (Div)
    std::vector<Expr> args_node_679 = {node_678, node_677};
    Call node_679(Op::Get("divide"), args_node_679, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.5/attention/Cast_1 (Cast)
    std::vector<Expr> args_node_680 = {node_679};
    Call node_680(Op::Get("cast"), args_node_680, CastAttrs::Create(1));
    // Node: /distilbert/transformer/layer.5/attention/Transpose_2 (Transpose)
    std::vector<Expr> args_node_681 = {node_667};
    Call node_681(Op::Get("transpose"), args_node_681, TransposeAttrs::Create({0, 2, 3, 1}));
    // Node: /distilbert/transformer/layer.5/attention/Sqrt_1 (Sqrt)
    std::vector<Expr> args_node_682 = {node_680};
    Call node_682(Op::Get("sqrt"), args_node_682, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Mul (Mul)
    std::vector<Expr> args_node_683 = {node_664, node_682};
    Call node_683(Op::Get("mul"), args_node_683, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Sqrt_2 (Sqrt)
    std::vector<Expr> args_node_684 = {node_680};
    Call node_684(Op::Get("sqrt"), args_node_684, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Mul_1 (Mul)
    std::vector<Expr> args_node_685 = {node_681, node_684};
    Call node_685(Op::Get("mul"), args_node_685, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/MatMul (MatMul)
    std::vector<Expr> args_node_686 = {node_683, node_685};
    Call node_686(Op::Get("matmul"), args_node_686, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Add (Add)
    std::vector<Expr> args_node_687 = {node_686, node_164};
    Call node_687(Op::Get("add"), args_node_687, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Softmax (Softmax)
    std::vector<Expr> args_node_688 = {node_687};
    Call node_688(Op::Get("softmax"), args_node_688, SoftmaxAttrs::Create(-1));
    // Node: /distilbert/transformer/layer.5/attention/MatMul_1 (MatMul)
    std::vector<Expr> args_node_689 = {node_688, node_671};
    Call node_689(Op::Get("matmul"), args_node_689, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Transpose_3 (Transpose)
    std::vector<Expr> args_node_690 = {node_689};
    Call node_690(Op::Get("transpose"), args_node_690, TransposeAttrs::Create({0, 2, 1, 3}));
    // Node: Constant_767 (Constant)
    Tensor node_691_tensor({1}, "int64");
    Constant node_691(node_691_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Unsqueeze_3 (Unsqueeze)
    std::vector<Expr> args_node_692 = {node_642, node_691};
    Call node_692(Op::Get("unsqueeze"), args_node_692, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/Constant_13 (Constant)
    Tensor node_693_tensor({1}, "int64");
    Constant node_693(node_693_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Constant_14 (Constant)
    Tensor node_694_tensor({1}, "int64");
    Constant node_694(node_694_tensor);
    // Node: /distilbert/transformer/layer.5/attention/Concat_3 (Concat)
    std::vector<Expr> args_node_695 = {node_692, node_693, node_694};
    Call node_695(Op::Get("concatenate"), args_node_695, ConcatAttrs::Create(0));
    // Node: /distilbert/transformer/layer.5/attention/Reshape_3 (Reshape)
    std::vector<Expr> args_node_696 = {node_690, node_695};
    Call node_696(Op::Get("reshape"), args_node_696, ReshapeAttrs::Create(0));
    // Node: /distilbert/transformer/layer.5/attention/out_lin/MatMul (MatMul)
    std::vector<Expr> args_node_697 = {node_696, const_98};
    Call node_697(Op::Get("matmul"), args_node_697, ObjectRef());
    // Node: /distilbert/transformer/layer.5/attention/out_lin/Add (Add)
    std::vector<Expr> args_node_698 = {const_57, node_697};
    Call node_698(Op::Get("add"), args_node_698, ObjectRef());
    // Node: /distilbert/transformer/layer.5/Add (Add)
    std::vector<Expr> args_node_699 = {node_698, node_639};
    Call node_699(Op::Get("add"), args_node_699, ObjectRef());
    // Node: /distilbert/transformer/layer.5/sa_layer_norm/ReduceMean (ReduceMean)
    std::vector<Expr> args_node_700 = {node_699};
    Call node_700(Op::Get("reduce_mean"), args_node_700, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.5/sa_layer_norm/Sub (Sub)
    std::vector<Expr> args_node_701 = {node_699, node_700};
    Call node_701(Op::Get("sub"), args_node_701, ObjectRef());
    // Node: /distilbert/transformer/layer.5/sa_layer_norm/Constant (Constant)
    Tensor node_702_tensor({}, "float32");
    Constant node_702(node_702_tensor);
    // Node: /distilbert/transformer/layer.5/sa_layer_norm/Pow (Pow)
    std::vector<Expr> args_node_703 = {node_701, node_702};
    Call node_703(Op::Get("pow"), args_node_703, ObjectRef());
    // Node: /distilbert/transformer/layer.5/sa_layer_norm/ReduceMean_1 (ReduceMean)
    std::vector<Expr> args_node_704 = {node_703};
    Call node_704(Op::Get("reduce_mean"), args_node_704, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.5/sa_layer_norm/Constant_1 (Constant)
    Tensor node_705_tensor({}, "float32");
    Constant node_705(node_705_tensor);
    // Node: /distilbert/transformer/layer.5/sa_layer_norm/Add (Add)
    std::vector<Expr> args_node_706 = {node_704, node_705};
    Call node_706(Op::Get("add"), args_node_706, ObjectRef());
    // Node: /distilbert/transformer/layer.5/sa_layer_norm/Sqrt (Sqrt)
    std::vector<Expr> args_node_707 = {node_706};
    Call node_707(Op::Get("sqrt"), args_node_707, ObjectRef());
    // Node: /distilbert/transformer/layer.5/sa_layer_norm/Div (Div)
    std::vector<Expr> args_node_708 = {node_701, node_707};
    Call node_708(Op::Get("divide"), args_node_708, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.5/sa_layer_norm/Mul (Mul)
    std::vector<Expr> args_node_709 = {node_708, const_58};
    Call node_709(Op::Get("mul"), args_node_709, ObjectRef());
    // Node: /distilbert/transformer/layer.5/sa_layer_norm/Add_1 (Add)
    std::vector<Expr> args_node_710 = {node_709, const_59};
    Call node_710(Op::Get("add"), args_node_710, ObjectRef());
    // Node: /distilbert/transformer/layer.5/ffn/lin1/MatMul (MatMul)
    std::vector<Expr> args_node_711 = {node_710, const_99};
    Call node_711(Op::Get("matmul"), args_node_711, ObjectRef());
    // Node: /distilbert/transformer/layer.5/ffn/lin1/Add (Add)
    std::vector<Expr> args_node_712 = {const_60, node_711};
    Call node_712(Op::Get("add"), args_node_712, ObjectRef());
    // Node: /distilbert/transformer/layer.5/ffn/activation/Constant (Constant)
    Tensor node_713_tensor({}, "float32");
    Constant node_713(node_713_tensor);
    // Node: /distilbert/transformer/layer.5/ffn/activation/Div (Div)
    std::vector<Expr> args_node_714 = {node_712, node_713};
    Call node_714(Op::Get("divide"), args_node_714, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.5/ffn/activation/Erf (Erf)
    std::vector<Expr> args_node_715 = {node_714};
    Call node_715(Op::Get("erf"), args_node_715, ErfAttrs::Create());
    // Node: /distilbert/transformer/layer.5/ffn/activation/Constant_1 (Constant)
    Tensor node_716_tensor({}, "float32");
    Constant node_716(node_716_tensor);
    // Node: /distilbert/transformer/layer.5/ffn/activation/Add (Add)
    std::vector<Expr> args_node_717 = {node_715, node_716};
    Call node_717(Op::Get("add"), args_node_717, ObjectRef());
    // Node: /distilbert/transformer/layer.5/ffn/activation/Mul (Mul)
    std::vector<Expr> args_node_718 = {node_712, node_717};
    Call node_718(Op::Get("mul"), args_node_718, ObjectRef());
    // Node: /distilbert/transformer/layer.5/ffn/activation/Constant_2 (Constant)
    Tensor node_719_tensor({}, "float32");
    Constant node_719(node_719_tensor);
    // Node: /distilbert/transformer/layer.5/ffn/activation/Mul_1 (Mul)
    std::vector<Expr> args_node_720 = {node_718, node_719};
    Call node_720(Op::Get("mul"), args_node_720, ObjectRef());
    // Node: /distilbert/transformer/layer.5/ffn/lin2/MatMul (MatMul)
    std::vector<Expr> args_node_721 = {node_720, const_100};
    Call node_721(Op::Get("matmul"), args_node_721, ObjectRef());
    // Node: /distilbert/transformer/layer.5/ffn/lin2/Add (Add)
    std::vector<Expr> args_node_722 = {const_61, node_721};
    Call node_722(Op::Get("add"), args_node_722, ObjectRef());
    // Node: /distilbert/transformer/layer.5/Add_1 (Add)
    std::vector<Expr> args_node_723 = {node_722, node_710};
    Call node_723(Op::Get("add"), args_node_723, ObjectRef());
    // Node: /distilbert/transformer/layer.5/output_layer_norm/ReduceMean (ReduceMean)
    std::vector<Expr> args_node_724 = {node_723};
    Call node_724(Op::Get("reduce_mean"), args_node_724, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.5/output_layer_norm/Sub (Sub)
    std::vector<Expr> args_node_725 = {node_723, node_724};
    Call node_725(Op::Get("sub"), args_node_725, ObjectRef());
    // Node: /distilbert/transformer/layer.5/output_layer_norm/Constant (Constant)
    Tensor node_726_tensor({}, "float32");
    Constant node_726(node_726_tensor);
    // Node: /distilbert/transformer/layer.5/output_layer_norm/Pow (Pow)
    std::vector<Expr> args_node_727 = {node_725, node_726};
    Call node_727(Op::Get("pow"), args_node_727, ObjectRef());
    // Node: /distilbert/transformer/layer.5/output_layer_norm/ReduceMean_1 (ReduceMean)
    std::vector<Expr> args_node_728 = {node_727};
    Call node_728(Op::Get("reduce_mean"), args_node_728, ReduceMeanAttrs::Create({-1}, 1));
    // Node: /distilbert/transformer/layer.5/output_layer_norm/Constant_1 (Constant)
    Tensor node_729_tensor({}, "float32");
    Constant node_729(node_729_tensor);
    // Node: /distilbert/transformer/layer.5/output_layer_norm/Add (Add)
    std::vector<Expr> args_node_730 = {node_728, node_729};
    Call node_730(Op::Get("add"), args_node_730, ObjectRef());
    // Node: /distilbert/transformer/layer.5/output_layer_norm/Sqrt (Sqrt)
    std::vector<Expr> args_node_731 = {node_730};
    Call node_731(Op::Get("sqrt"), args_node_731, ObjectRef());
    // Node: /distilbert/transformer/layer.5/output_layer_norm/Div (Div)
    std::vector<Expr> args_node_732 = {node_725, node_731};
    Call node_732(Op::Get("divide"), args_node_732, DivAttrs::Create());
    // Node: /distilbert/transformer/layer.5/output_layer_norm/Mul (Mul)
    std::vector<Expr> args_node_733 = {node_732, const_62};
    Call node_733(Op::Get("mul"), args_node_733, ObjectRef());
    // Node: /distilbert/transformer/layer.5/output_layer_norm/Add_1 (Add)
    std::vector<Expr> args_node_734 = {node_733, const_63};
    Call node_734(Op::Get("add"), args_node_734, ObjectRef());
    // Node: /qa_outputs/MatMul (MatMul)
    std::vector<Expr> args_node_735 = {node_734, const_101};
    Call node_735(Op::Get("matmul"), args_node_735, ObjectRef());
    // Node: /qa_outputs/Add (Add)
    std::vector<Expr> args_node_736 = {const_64, node_735};
    Call node_736(Op::Get("add"), args_node_736, ObjectRef());
    // Node: /Constant (Constant)
    Tensor node_737_tensor({2}, "int64");
    Constant node_737(node_737_tensor);
    // Node: /Split (Split)
    std::vector<Expr> args_node_738 = {node_736, node_737};
    Call node_738(Op::Get("split"), args_node_738, SplitAttrs::Create(-1));
    // Node: /Constant_1 (Constant)
    Tensor node_739_tensor({1}, "int64");
    Constant node_739(node_739_tensor);
    // Node: /Squeeze (Squeeze)
    std::vector<Expr> args_node_740 = {node_738, node_739};
    Call node_740(Op::Get("squeeze"), args_node_740, ObjectRef());
    // Node: /Constant_2 (Constant)
    Tensor node_741_tensor({1}, "int64");
    Constant node_741(node_741_tensor);
    // Node: /Squeeze_1 (Squeeze)
    std::vector<Expr> args_node_742 = {node_738, node_741};
    Call node_742(Op::Get("squeeze"), args_node_742, ObjectRef());

    // --- Output ---
    return node_740;
}

int main() {
    try {
        Expr graph = build_graph();
        std::cout << "Graph built successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}