// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 George Rennie

#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>

extern "C" {
#include "aiger.h"
}

namespace Aiger {

// Type aliases for aiger and variable types
using Lit = decltype(aiger_symbol::lit);
using Var = decltype(aiger::maxvar);
static_assert(std::is_same_v<Lit, Var>);

// We want to be able to assume that lit is an unsigned int of at least 32 bits
// in our code
static_assert(std::numeric_limits<Lit>::is_integer);
static_assert(!std::numeric_limits<Lit>::is_signed);
static_assert(std::numeric_limits<Lit>::digits >= 32);

// Helper for managing aiger with RAII
using Aig = std::shared_ptr<aiger>;
using ConstAig = std::shared_ptr<const aiger>;
inline Aig make_aig() { return {aiger_init(), &aiger_reset}; }

// Return non-zero on success on success
Aig checked_read(const std::string& file);
bool checked_write(Aig aig, const std::string& file);

bool is_combinational(const ConstAig& aig);
bool has_properties(const ConstAig& aig);

enum class SymbType : uint8_t { Input, Latch, Output, Bad, Constraint, Justice, Fairness };

// Copy an aig, renaming all symbols using the renaming function func
Aig rename(
    ConstAig aig,
    std::function<std::optional<std::string>(const aiger_symbol& symb, const SymbType type)> func
);

// Perform basic optimisations (constant propagation, single-level strashing
// and COI reduction) on an (potentially sequential) aig. Note that the only
// information this keeps invariant is the set of inputs/outputs/bad/constraints
// and the names on latches that remain. latches/ands can be removed and their
// literals changed
Aig strash(ConstAig unoptimised);

// clang-format off
inline std::span<aiger_symbol> inputs(const Aig& aig)      { return {aig->inputs,      aig->num_inputs}; }
inline std::span<aiger_symbol> latches(const Aig& aig)     { return {aig->latches,     aig->num_latches}; }
inline std::span<aiger_symbol> outputs(const Aig& aig)     { return {aig->outputs,     aig->num_outputs}; }
inline std::span<aiger_symbol> bads(const Aig& aig)        { return {aig->bad,         aig->num_bad}; }
inline std::span<aiger_symbol> constraints(const Aig& aig) { return {aig->constraints, aig->num_constraints}; }
inline std::span<aiger_symbol> justices(const Aig& aig)    { return {aig->justice,     aig->num_justice}; }
inline std::span<aiger_symbol> fairnesses(const Aig& aig)  { return {aig->fairness,    aig->num_fairness}; }
inline std::span<aiger_and>    gates(const Aig& aig)       { return {aig->ands,        aig->num_ands}; }

inline std::span<const aiger_symbol> inputs(const ConstAig& aig)      { return {aig->inputs,      aig->num_inputs}; }
inline std::span<const aiger_symbol> latches(const ConstAig& aig)     { return {aig->latches,     aig->num_latches}; }
inline std::span<const aiger_symbol> outputs(const ConstAig& aig)     { return {aig->outputs,     aig->num_outputs}; }
inline std::span<const aiger_symbol> bads(const ConstAig& aig)        { return {aig->bad,         aig->num_bad}; }
inline std::span<const aiger_symbol> constraints(const ConstAig& aig) { return {aig->constraints, aig->num_constraints}; }
inline std::span<const aiger_symbol> justices(const ConstAig& aig)    { return {aig->justice,     aig->num_justice}; }
inline std::span<const aiger_symbol> fairnesses(const ConstAig& aig)  { return {aig->fairness,    aig->num_fairness}; }
inline std::span<const aiger_and>    gates(const ConstAig& aig)       { return {aig->ands,        aig->num_ands}; }

inline std::span<Lit> justice_lits(const aiger_symbol& justice) { return {justice.lits, justice.size}; }
// clang-format on

[[nodiscard]] inline Lit next_lit(ConstAig aig) { return aiger_var2lit(aig->maxvar + 1); }

// Add and gate, performing basic constant propagation/simplification
// Doesn't support fairness or justice nodes
[[nodiscard]] inline Lit add_and(Aig aig, const Lit rhs0, const Lit rhs1) {
	if (rhs0 == aiger_false || rhs1 == aiger_false || rhs0 == aiger_not(rhs1))
		return aiger_false;

	if (rhs0 == aiger_true || rhs0 == rhs1)
		return rhs1;

	if (rhs1 == aiger_true)
		return rhs0;

	const auto lhs = next_lit(aig);
	aiger_add_and(aig.get(), lhs, rhs0, rhs1);
	return lhs;
}

[[nodiscard]] inline Lit add_or(Aig aig, const Lit rhs0, const Lit rhs1) {
	return aiger_not(add_and(aig, aiger_not(rhs0), aiger_not(rhs1)));
}

[[nodiscard]] inline Lit add_implies(Aig aig, const Lit pre, const Lit post) {
	return add_or(aig, aiger_not(pre), post);
}

[[nodiscard]] inline Lit add_bit_eq(Aig aig, const Lit a, const Lit b) {
	// a == b <==> a XNOR b <==> ¬((¬a /\ b) \/ (a /\ ¬b))
	// ¬(¬a /\ b) /\ ¬(a /\ ¬b))
	const auto left = add_and(aig, aiger_not(a), b);
	const auto right = add_and(aig, a, aiger_not(b));
	return add_and(aig, aiger_not(left), aiger_not(right));
}

inline Lit add_input(Aig aig, const char* name = nullptr) {
	const auto lit = next_lit(aig);
	aiger_add_input(aig.get(), lit, name);
	return lit;
}

inline Lit add_latch(
    Aig aig, const Lit next, const char* name = nullptr, std::optional<Lit> reset = aiger_false
) {
	const auto lit = next_lit(aig);
	aiger_add_latch(aig.get(), lit, next, name);
	assert(!reset || aiger_is_constant(*reset));
	aiger_add_reset(aig.get(), lit, reset ? *reset : lit);
	return lit;
}

} // namespace Aiger
