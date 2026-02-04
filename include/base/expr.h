#pragma once
#include "object.h"
#include <string>

namespace kxc {

class SpanNode : public Object {
public:
    std::string source_name;
    int line;
    int column;
    
    const TypeIndex GetTypeId() const override {
        return kKXC_OBJECT_TYPE + 2; 
    }
};

class Span : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    Span(std::string source_name, int line, int column) {
        auto* node = new SpanNode();
        node->source_name = std::move(source_name);
        node->line = line;
        node->column = column;
        SetData(node);
    }
};

class ExprNode : public Object {
public:
    Span span;
    // DataType dtype; // Placeholder for data type

    const TypeIndex GetTypeId() const override { 
        return kKXC_OBJECT_TYPE + 3; 
    }
    virtual bool StructualEqual(const ExprNode* other) const {
        return this->GetTypeId() == other->GetTypeId();
    }
    virtual size_t StructuralHash() const {
        return std::hash<TypeIndex>()(this->GetTypeId());
    }
};

class Expr : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
};

}
