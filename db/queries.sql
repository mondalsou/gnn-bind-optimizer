-- GNNBindOptimizer — example analytical queries
USE gnnbind;
GO

-- 1. Best RL molecule per experiment (highest reward)
SELECT
    e.name            AS experiment,
    rl.smiles,
    rl.reward         AS best_reward,
    rl.pred_pkd,
    rl.r_qed,
    rl.r_sa,
    rl.step
FROM dbo.rl_molecules rl
INNER JOIN dbo.experiments e ON e.id = rl.experiment_id
WHERE rl.reward = (
    SELECT MAX(r2.reward)
    FROM dbo.rl_molecules r2
    WHERE r2.experiment_id = rl.experiment_id
)
ORDER BY rl.reward DESC;
GO

-- 2. MTL vs STL ablation comparison (best val_rmse per model type)
SELECT
    e.name        AS experiment,
    mr.model_type,
    MIN(mr.val_rmse)      AS best_val_rmse,
    MAX(mr.val_pearson_r) AS best_pearson_r,
    MAX(mr.val_auc_pose)  AS best_pose_auc
FROM dbo.model_runs mr
INNER JOIN dbo.experiments e ON e.id = mr.experiment_id
GROUP BY e.name, mr.model_type
ORDER BY best_val_rmse ASC;
GO

-- 3. RL reward trajectory — moving average over 20 steps
SELECT
    step,
    reward,
    AVG(reward) OVER (
        ORDER BY step
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) AS reward_ma10,
    pred_pkd
FROM dbo.rl_molecules
WHERE experiment_id = (
    SELECT id FROM dbo.experiments WHERE name = 'rl_reinforce_egfr'
)
ORDER BY step;
GO

-- 4. Drug-likeness distribution — QED/SA/MW buckets for RL molecules
SELECT
    CASE
        WHEN r_qed >= 0.7 THEN 'high'
        WHEN r_qed >= 0.4 THEN 'mid'
        ELSE 'low'
    END AS qed_bucket,
    CASE
        WHEN r_sa >= 0.7 THEN 'easy'
        WHEN r_sa >= 0.5 THEN 'moderate'
        ELSE 'hard'
    END AS sa_bucket,
    COUNT(*) AS mol_count,
    AVG(pred_pkd) AS mean_pkd,
    MAX(reward)   AS max_reward
FROM dbo.rl_molecules
WHERE experiment_id = (
    SELECT id FROM dbo.experiments WHERE name = 'rl_reinforce_egfr'
)
GROUP BY
    CASE WHEN r_qed >= 0.7 THEN 'high' WHEN r_qed >= 0.4 THEN 'mid' ELSE 'low' END,
    CASE WHEN r_sa  >= 0.7 THEN 'easy' WHEN r_sa  >= 0.5 THEN 'moderate' ELSE 'hard' END
ORDER BY mean_pkd DESC;
GO

-- 5. GNN vs Vina parity — correlation on benchmark set
SELECT
    vb.pocket_pdb,
    vb.smiles,
    vb.vina_score,
    vb.gnn_pred_pkd,
    ABS(vb.vina_score - vb.gnn_pred_pkd) AS abs_error
FROM dbo.vina_benchmarks vb
ORDER BY abs_error DESC;
GO

-- 6. Top 20 binding predictions across all runs (by pred_pkd)
SELECT TOP 20
    e.name        AS experiment,
    bp.smiles,
    bp.pocket_pdb,
    bp.pred_pkd,
    bp.pred_pose_prob,
    bp.pred_select_prob,
    bp.created_at
FROM dbo.binding_predictions bp
INNER JOIN dbo.model_runs mr ON mr.id = bp.run_id
INNER JOIN dbo.experiments e ON e.id = mr.experiment_id
ORDER BY bp.pred_pkd DESC;
GO

-- 7. Pareto front: high affinity AND high drug-likeness (reward > 0.65 AND pred_pkd > 7.0)
SELECT
    smiles,
    reward,
    pred_pkd,
    r_qed,
    r_sa,
    r_mw,
    step
FROM dbo.rl_molecules
WHERE reward > 0.65 AND pred_pkd > 7.0
ORDER BY reward DESC, pred_pkd DESC;
GO
