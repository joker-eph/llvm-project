# Registered Properties

MLIR has two forms of operation property values. Attributes are immutable and
owned by an `MLIRContext`. Native properties retain their C++ storage and are
owned by an operation or an `OwningProperty` value. A registered native property
kind provides a semantic `TypeID`, a dialect-qualified name, intrinsic
verification, parsing, printing, and value operations. Its identity is separate
from the C++ type used for storage, so two kinds may both store `bool` while
remaining distinct.

`Property` is a borrowed read-only view of either form. `OwningProperty` owns a
standalone native value, or retains a context-owned attribute. A native value
is cloned explicitly with `OwningProperty::copy` or `clone`. The context must
outlive all native values and views. An operation must outlive its
`OperationPropertyRef` views.

Dialects register native kinds with `addProperties<Kinds...>()`. A kind supplies
`StorageType`, `name`, `verify`, `parse`, and `print`; custom `hash` is optional.
The registration metadata belongs to a TableGen `PropDef`. An operation field
refers to it through `propertyKind`, while the existing `Property` record keeps
the field's default, constraint, storage, and compatibility hooks. For example,
`DefaultValuedProp` and `ConfinedProp` preserve the same `PropDef` identity.

```tablegen
def CountKind : PropDef<"int64_t", "count", "test", "CountKind"> {
  let parser = [{ return parser.parseInteger(value); }];
  let printer = [{ printer.printInteger(value); }];
}
def CountField : RegisteredProp<CountKind>;
def PositiveCountField : ConfinedProp<CountField, CPred<"$_self > 0">>;
```

`ArrayPropertyKind<ElementKind>` and `OptionalPropertyKind<ElementKind>` are
composed C++ kinds. Each template instantiation has its own semantic `TypeID`,
including when two element kinds share a storage type. Register the element
kind before its compositions. Their names include the element's dialect and
mnemonic, for example `builtin.array.builtin.i64`. TableGen fields can use
`RegisteredArrayProp<ElementField>` and
`RegisteredOptionalProp<ElementField>`; the element field supplies existing ODS
syntax and bytecode hooks while its `PropDef` selects the composed kind.
Composed kind registration itself does not provide legacy attribute conversion;
the field's existing ODS conversion hooks remain available.

Lookup is available by semantic `TypeID` or qualified name. Native values
parse as `&dialect.mnemonic<payload>`, and a caller that already knows the kind
can parse just `payload`:

```c++
auto value = parseProperty("&builtin.i64<42>", context);
const AbstractProperty *kind =
    AbstractProperty::lookup("builtin.i64", context);
auto sameValue = parseProperty("42", *kind);
```

An operation may publish `PropertyFieldDescriptor` entries for its declared
fields. Descriptors use callbacks to read and write the operation's existing
storage layout. `Operation::getPropertyField` gives a checked borrowed
reference; `copy()` makes a standalone snapshot. Assignment verifies context,
semantic kind, intrinsic validity, and field-local constraints before writing.
Generated references can reset defaulted fields while required fields reject
reset.
`OperationState::setNamedProperty` copies a field into temporary typed storage
before result-type inference. An `InferTypeOpInterface` implementation can read
that typed storage without converting it to an attribute. Operations that
declare a complete field list
may opt in to generic native assembly such as:

```mlir
"python_test.properties"() <{count = &builtin.i64<42>}> : () -> ()
```

The C API owns `MlirProperty` values explicitly and destroys them with
`mlirPropertyDestroy`. `MlirOperationPropertyRef` borrows from its operation.
`mlirOperationCreateWithProperties` copies named values before inference; its
state attributes represent discardable attributes. The Python `Property`
object owns its C value and retains the context. `op.properties[name]` makes
an owned snapshot; `op.properties.ref(name)` follows the live field and raises
after the operation is erased.

Existing attribute APIs and operation property storage remain available.
Legacy attribute encodings may still be parsed when an operation provides a
conversion hook. Operations without complete field descriptors continue to
use their existing generic assembly format.

An operation whose native properties have no attribute conversion can implement
`BytecodeOpInterface` to serialize them directly. The Python test operation
`python_test.properties` uses a native i64 field to infer its result type and
deliberately rejects legacy attribute conversion. The test dialect exports a
shared read-only interface through its C API; it returns a number from either
an i64 property or an integer attribute.
