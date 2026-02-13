#pragma once
#include "object.h"
#include <string>

namespace kxc {

class TypeNode : public Object {
public:
    KXC_OBJECT_DECLARE
};

class Type : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    const TypeNode* operator->() const { return static_cast<const TypeNode*>(object_); }
};

class SpanNode : public Object {
public:
    std::string source_name;
    int line;
    int column;
    
    KXC_OBJECT_DECLARE
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
    Type checked_type_; 

    KXC_OBJECT_DECLARE
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
    Expr(ObjectRef n) : ObjectRef(n) {}
    
    Type checked_type() const {
        return static_cast<const ExprNode*>(object_)->checked_type_;
    }
};

}
