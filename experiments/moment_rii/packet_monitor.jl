# Lean-facing packet monitor utilities for moment-NLFEAST experiments.
#
# These stay in the experiment layer. They encode the diagnostic boundary from
# LEAN_FEAST/ALGORITHM_HANDOFF.md without promoting a public package API.

function oblique_packet_projector(right, left; ranktol=1e-10)
    X, _ = physical_basis_from_columns(right; ranktol=ranktol)
    Y, _ = physical_basis_from_columns(left; ranktol=ranktol)
    d = min(size(X, 2), size(Y, 2))
    d == 0 && return zeros(ComplexF64, size(right, 1), size(right, 1))
    X = X[:, 1:d]
    Y = Y[:, 1:d]
    X * pinv(adjoint(Y) * X) * adjoint(Y)
end

function projector_defect_split(Pcandidate, Preference)
    defect = Pcandidate - Preference
    defect_norm = norm(defect)
    if defect_norm <= eps(Float64)
        return (total=0.0, visible=0.0, invisible=0.0, visible_ratio=0.0, invisible_ratio=0.0)
    end
    visible = norm(Preference * defect)
    invisible = norm((I - Preference) * defect)
    (
        total=defect_norm,
        visible=visible,
        invisible=invisible,
        visible_ratio=visible / defect_norm,
        invisible_ratio=invisible / defect_norm,
    )
end

function packet_membership_gap(columns, basis; ranktol=1e-10)
    norm(columns) <= eps(Float64) && return 0.0
    Q, _ = physical_basis_from_columns(basis; ranktol=ranktol)
    size(Q, 2) == 0 && return 1.0
    norm(columns - Q * (adjoint(Q) * columns)) / norm(columns)
end

function packet_trial_test_compatibility(right, left, right_trial, left_test; ranktol=1e-10)
    right_gap = packet_membership_gap(right, right_trial; ranktol=ranktol)
    left_gap = packet_membership_gap(left, left_test; ranktol=ranktol)
    (
        right_gap=right_gap,
        left_gap=left_gap,
        max_gap=max(right_gap, left_gap),
    )
end

function packet_schedule_proxy(Praw, Prefined, Preference)
    raw_defect = Praw - Preference
    refined_defect = Prefined - Preference
    repair = Prefined - Praw
    Qreference = I - Preference
    raw_visible = Preference * raw_defect
    repair_visible = Preference * repair
    (
        q_current=norm(Qreference * Praw),
        q_error=norm(Qreference * raw_defect),
        q_repair=norm(Qreference * repair),
        visible_error=norm(raw_visible),
        visible_repair=norm(repair_visible),
        observed_repair=norm(repair),
        # Lean predicts the scheduled repair is biased by -P_packet(E current).
        visible_bias_residual=norm(repair_visible + raw_visible),
        visible_contraction=norm(Preference * refined_defect) / max(norm(raw_visible), eps(Float64)),
        correction_coordinate_relative_error=norm(repair_visible + raw_visible) / max(norm(raw_visible), eps(Float64)),
        refined_visible_error=norm(Preference * refined_defect),
    )
end

function packet_schedule_monitor(schedule; inner_steps, feast_ratio=nothing, outer_index=1, contraction=schedule.visible_contraction)
    inner_multiplier = feast_ratio === nothing ? nothing : feast_ratio^inner_steps
    inner_budget = inner_multiplier === nothing ? nothing : inner_multiplier * (schedule.q_current + schedule.q_error)
    outer_budget = contraction^outer_index * schedule.visible_error
    (
        biterate=schedule.q_current,
        brepair=schedule.q_error,
        inner_steps=inner_steps,
        feast_ratio=feast_ratio,
        inner_budget=inner_budget,
        outer_index=outer_index,
        contraction=contraction,
        outer_visible_budget=outer_budget,
        observed_scheduled_repair=schedule.observed_repair,
        observed_total_budget=inner_budget === nothing ? outer_budget : inner_budget + outer_budget,
    )
end

function packet_defect_status(row, expected; visible_tol=1e-8)
    if row.refined_matched == expected && row.refined_defect.visible <= visible_tol
        :accepted_visible_removed
    elseif row.refined_defect.visible > visible_tol
        :packet_visible_defect
    else
        :packet_invisible_acceptance_gap
    end
end

function packet_acceptance_certificate(row, expected; visible_tol=1e-8, contraction_tol=1e-7)
    rank_ok = row.rank == expected
    membership_ok = row.refined_matched == expected
    visible_ok = row.refined_defect.visible <= visible_tol
    contraction_ok = row.schedule.visible_contraction <= contraction_tol
    correction_coordinate_ok = row.schedule.correction_coordinate_relative_error <= contraction_tol
    residual_ok = row.refined_good == expected
    accepted = rank_ok && membership_ok && visible_ok && contraction_ok && correction_coordinate_ok && residual_ok
    (
        accepted=accepted,
        rank_ok=rank_ok,
        membership_ok=membership_ok,
        visible_ok=visible_ok,
        contraction_ok=contraction_ok,
        correction_coordinate_ok=correction_coordinate_ok,
        residual_ok=residual_ok,
        score=count((rank_ok, membership_ok, visible_ok, contraction_ok, correction_coordinate_ok, residual_ok)),
        total=6,
    )
end

function packet_defect_policy_action(status, acceptance)
    acceptance.accepted && return :accept
    status === :accepted_visible_removed && return :accept
    status === :packet_visible_defect && return :increase_nodes_or_refine_chart
    status === :packet_invisible_acceptance_gap && return :refine_extraction_or_acceptance
    :inspect
end

function packet_update_stage(row)
    if row.acceptance.accepted
        return :accept
    elseif !row.acceptance.visible_ok || row.schedule.visible_contraction > 0.5
        return :rebuild_packet_update
    elseif !row.acceptance.contraction_ok || !row.acceptance.correction_coordinate_ok
        return :continue_local_repair_schedule
    elseif !row.acceptance.membership_ok || !row.acceptance.residual_ok
        return :improve_reduced_extraction_acceptance
    else
        return :inspect
    end
end

function merged_packet_action(actions)
    :increase_nodes_or_refine_chart in actions && return :increase_nodes_or_refine_chart
    :refine_extraction_or_acceptance in actions && return :refine_extraction_or_acceptance
    :accept in actions && all(==(:accept), actions) && return :accept
    :inspect
end

function merged_packet_update_stage(stages)
    :rebuild_packet_update in stages && return :rebuild_packet_update
    :continue_local_repair_schedule in stages && return :continue_local_repair_schedule
    :improve_reduced_extraction_acceptance in stages && return :improve_reduced_extraction_acceptance
    :accept in stages && all(==(:accept), stages) && return :accept
    :inspect
end

function merge_packet_policy_reports(rows)
    actions = Tuple(row.action for row in rows)
    stages = Tuple(row.update_stage for row in rows)
    accepted = all(row.acceptance.accepted for row in rows)
    scores = sum(row.acceptance.score for row in rows)
    totals = sum(row.acceptance.total for row in rows)
    (
        accepted=accepted,
        action=accepted ? :accept : merged_packet_action(actions),
        update_stage=accepted ? :accept : merged_packet_update_stage(stages),
        component_actions=actions,
        component_update_stages=stages,
        component_statuses=Tuple(row.status for row in rows),
        score=scores,
        total=totals,
    )
end

function packet_steering_trace(rows)
    accepted_index = findfirst(row -> row.acceptance.accepted, rows)
    selected = accepted_index === nothing ? nothing : rows[accepted_index]
    trace = Tuple(
        (
            action=row.action,
            update_stage=row.update_stage,
            accepted=row.acceptance.accepted,
            score=row.acceptance.score,
            total=row.acceptance.total,
        ) for row in rows
    )
    (
        accepted=selected !== nothing,
        selected_index=accepted_index,
        selected=selected,
        final_action=selected === nothing ? rows[end].action : :accept,
        final_update_stage=selected === nothing ? rows[end].update_stage : :accept,
        trace=trace,
    )
end

function packet_chart_ladder_trace(rows, target_expected)
    same_packet_index = findfirst(row -> row.expected == target_expected && row.acceptance.accepted, rows)
    accepted_index = findfirst(row -> row.acceptance.accepted, rows)
    changed_packet_index = findfirst(row -> row.expected != target_expected && row.acceptance.accepted, rows)
    if same_packet_index !== nothing
        action = :accept_same_packet
        selected = rows[same_packet_index]
    elseif changed_packet_index !== nothing
        action = :chart_changes_packet
        selected = rows[changed_packet_index]
    elseif any(row -> row.status === :packet_invisible_acceptance_gap, rows)
        action = :refine_extraction_or_acceptance
        selected = nothing
    else
        action = :increase_nodes_or_refine_chart
        selected = nothing
    end
    (
        action=action,
        accepted_same_packet=same_packet_index !== nothing,
        accepted_any_packet=accepted_index !== nothing,
        selected=selected,
        selected_index=selected === nothing ? nothing : findfirst(==(selected), rows),
        target_expected=target_expected,
        component_expected=Tuple(row.expected for row in rows),
        component_statuses=Tuple(row.status for row in rows),
    )
end
