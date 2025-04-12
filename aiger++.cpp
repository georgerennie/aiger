// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 George Rennie

#include <fmt/core.h>
#include <algorithm>
#include <cassert>
#include <stack>
#include <vector>
#include "aiger++.hpp"

namespace Aiger {

Aig checked_read(const std::string& file) {
	auto aig = make_aig();

	if (const auto aig_err = file == "-" ? aiger_read_from_file(aig.get(), stdin)
	                                     : aiger_open_and_read_from_file(aig.get(), file.c_str())) {
		fmt::println(stderr, R"(ERROR: error reading aiger file "{}":)", file);
		fmt::println(stderr, "{}", aig_err);
		return nullptr;
	}

	if (const auto aig_err = aiger_check(aig.get())) {
		fmt::println(stderr, R"(ERROR: aiger file "{}" is invalid:)", file);
		fmt::println(stderr, "{}", aig_err);
		return nullptr;
	}

	return aig;
}

bool checked_write(Aig aig, const std::string& file) {
	if (const auto aig_err = aiger_check(aig.get())) {
		fmt::println(stderr, R"(ERROR: aiger is invalid (before writing to "{}"):)", file);
		fmt::println(stderr, "{}", aig_err);
		return false;
	}

	if (!(file == "-" ? aiger_write_to_file(aig.get(), aiger_ascii_mode, stdout)
	                  : aiger_open_and_write_to_file(aig.get(), file.c_str()))) {
		fmt::println(stderr, R"(ERROR: error writing aiger file "{}")", file);
		return false;
	}

	return true;
}

bool is_combinational(const ConstAig& aig) { return aig->latches == 0; }

bool has_properties(const ConstAig& aig) {
	return (aig->num_bad > 0) || (aig->num_constraints > 0) || (aig->num_justice > 0) ||
	       (aig->num_fairness > 0);
}

Aig rename(
    ConstAig aig,
    std::function<std::optional<std::string>(const aiger_symbol& symb, const SymbType type)> func
) {
	auto renamed = make_aig();

	const auto name = [&func](const aiger_symbol& symb, const SymbType type) -> const char* {
		if (const auto name = func(symb, type))
			return name->c_str();
		return nullptr;
	};

	// Copy simple nodes
	for (const auto& input : inputs(aig))
		aiger_add_input(renamed.get(), input.lit, name(input, SymbType::Input));
	for (const auto& output : outputs(aig))
		aiger_add_output(renamed.get(), output.lit, name(output, SymbType::Output));
	for (const auto& bad : bads(aig))
		aiger_add_bad(renamed.get(), bad.lit, name(bad, SymbType::Bad));
	for (const auto& constraint : constraints(aig))
		aiger_add_constraint(renamed.get(), constraint.lit, name(constraint, SymbType::Constraint));
	for (const auto& justice : justices(aig))
		aiger_add_justice(
		    renamed.get(), justice.size, justice.lits, name(justice, SymbType::Justice)
		);
	for (const auto& fairness : fairnesses(aig))
		aiger_add_constraint(renamed.get(), fairness.lit, name(fairness, SymbType::Fairness));

	// Copy latches
	for (const auto& latch : latches(aig)) {
		aiger_add_latch(renamed.get(), latch.lit, latch.next, name(latch, SymbType::Latch));
		aiger_add_reset(renamed.get(), latch.lit, latch.reset);
	}

	// Copy gates
	for (const auto& gate : gates(aig))
		aiger_add_and(renamed.get(), gate.lhs, gate.rhs0, gate.rhs1);

	return renamed;
}

Aig strash(ConstAig unoptimised) {
	auto aig = make_aig();

	// A map from variables in the unoptimised AIG to literals in the optimised
	// AIG. They are defaulted to the Lit::max() to enable error checking,
	// making sure that each variable that gets used has a map
	constexpr Lit null_lit = std::numeric_limits<Lit>::max();
	std::vector<Lit> var_map(unoptimised->maxvar + 1, null_lit);
	var_map[aiger_lit2var(aiger_false)] = aiger_strip(aiger_false);

	Var var_idx = 1;
	const auto new_lit = [&]() -> Lit { return aiger_var2lit(var_idx++); };
	const auto mapped = [&](const Lit lit) -> bool {
		return var_map.at(aiger_lit2var(lit)) != null_lit;
	};
	const auto map = [&](const Lit lit) -> Lit {
		return var_map.at(aiger_lit2var(lit)) ^ aiger_sign(lit);
	};
	const auto new_map = [&](const Lit lit, const Lit mapped_lit) -> Lit {
		var_map[aiger_lit2var(lit)] = mapped_lit;
		return mapped_lit;
	};

	// Do a depth first search back from the output node to combine COI red
	// and topo sort. For each node, make sure we map its children before it.
	std::stack<Lit> nodes_to_visit;

	// Returns true if AND gates have been enqueued for further processing, and
	// false if the node is (now) mapped
	const auto enqueue = [&](const Lit lit) -> bool {
		if (mapped(lit))
			return false;

		assert(aiger_is_and(unoptimised.get(), lit) || aiger_is_latch(unoptimised.get(), lit));
		nodes_to_visit.emplace(lit);
		return true;
	};

	// Add the inputs
	for (const auto& input : inputs(unoptimised))
		aiger_add_input(aig.get(), new_map(input.lit, new_lit()), input.name);

	// Add all the different output types' COIs
	for (const auto& output : outputs(unoptimised))
		enqueue(output.lit);
	for (const auto& bad : bads(unoptimised))
		enqueue(bad.lit);
	for (const auto& constraint : constraints(unoptimised))
		enqueue(constraint.lit);
	for (const auto& justice : justices(unoptimised))
		for (const auto lit : justice_lits(justice))
			enqueue(lit);
	for (const auto& fairness : fairnesses(unoptimised))
		enqueue(fairness.lit);

	// Store gates that have been seen before. The index corresponds to the min
	// of the two input literals, and the entries are pairs of the max of
	// the input literals and the output literal.
	using Entry = std::pair<Lit, Lit>;
	std::vector<std::vector<Entry>> known_gates(aiger_var2lit(unoptimised->maxvar + 1));

	// Loop through AND gates, visiting their children as necessary
	while (!nodes_to_visit.empty()) {
		if (const auto gate = aiger_is_and(unoptimised.get(), nodes_to_visit.top())) {
			if (enqueue(gate->rhs0) || enqueue(gate->rhs1))
				continue;

			// Both children have now been mapped, so we can map this node
			nodes_to_visit.pop();

			if (mapped(gate->lhs))
				continue;

			// Order children
			const auto orig_rhs0 = map(gate->rhs0);
			const auto orig_rhs1 = map(gate->rhs1);
			const auto rhs0 = std::min(orig_rhs0, orig_rhs1);
			const auto rhs1 = std::max(orig_rhs0, orig_rhs1);

			const auto& potential_matches = known_gates.at(rhs0);

			// Perform constant propagation and single level strashing
			// - false /\ a <==> false
			// - true /\ b <==> b
			// - a /\ ¬a <==> false
			// - a /\ a <==> a
			if (rhs0 == aiger_false || rhs1 == aiger_false || rhs0 == aiger_not(rhs1)) {
				new_map(gate->lhs, aiger_false);
			} else if (rhs0 == aiger_true || rhs0 == rhs1) {
				new_map(gate->lhs, rhs1);
			} else if (rhs1 == aiger_true) {
				new_map(gate->lhs, rhs0);
			} else if (const auto known = std::find_if(
			               potential_matches.cbegin(), potential_matches.cend(),
			               [rhs1](const auto& entry) { return entry.first == rhs1; }
			           );
			           known != potential_matches.cend()) {
				new_map(gate->lhs, known->second);
			} else {
				const auto gate_lit = new_lit();
				aiger_add_and(aig.get(), new_map(gate->lhs, gate_lit), rhs0, rhs1);
				known_gates.at(rhs0).emplace_back(rhs1, gate_lit);
			}
		} else if (const auto latch = aiger_is_latch(unoptimised.get(), nodes_to_visit.top())) {
			// Map a variable first so we don't visit this node again in an SCC
			if (!mapped(latch->lit))
				new_map(aiger_strip(latch->lit), new_lit());

			// Visit predecessor and resolve
			if (enqueue(latch->next))
				continue;

			// Predecessor has now been mapped so we can map this node
			nodes_to_visit.pop();

			const auto lit = map(latch->lit);

			// If we have already added the latch for this variable keep going
			if (aiger_lit2var(lit) <= aig->maxvar && aiger_is_latch(aig.get(), lit))
				continue;

			aiger_add_latch(aig.get(), lit, map(latch->next), latch->name);

			const auto reset = aiger_is_constant(latch->reset) ? latch->reset : lit;
			aiger_add_reset(aig.get(), lit, reset);
		}
	}

	// Add back outputs with transformed lits
	for (const auto& output : outputs(unoptimised))
		aiger_add_output(aig.get(), map(output.lit), output.name);
	for (const auto& bad : bads(unoptimised))
		aiger_add_bad(aig.get(), map(bad.lit), bad.name);
	for (const auto& constraint : constraints(unoptimised))
		aiger_add_constraint(aig.get(), map(constraint.lit), constraint.name);
	for (const auto& justice : justices(unoptimised)) {
		std::vector<Lit> new_lits;
		const auto old_lits = justice_lits(justice);
		new_lits.reserve(old_lits.size());
		std::transform(old_lits.begin(), old_lits.end(), std::back_inserter(new_lits), map);
		aiger_add_justice(aig.get(), new_lits.size(), new_lits.data(), justice.name);
	}
	for (const auto& fairness : fairnesses(unoptimised))
		aiger_add_constraint(aig.get(), map(fairness.lit), fairness.name);

	return aig;
}

} // namespace Aiger
