-- GNNBindOptimizer — SQL Server 2022 schema init
-- Run once on container startup via entrypoint

IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'gnnbind')
    CREATE DATABASE gnnbind;
GO

USE gnnbind;
GO

-- ─── experiments ────────────────────────────────────────────────────────────
IF OBJECT_ID('dbo.experiments', 'U') IS NULL
CREATE TABLE dbo.experiments (
    id           INT IDENTITY(1,1) PRIMARY KEY,
    name         NVARCHAR(256)  NOT NULL,
    created_at   DATETIME2      NOT NULL DEFAULT SYSDATETIME(),
    config_json  NVARCHAR(MAX)  NULL,
    CONSTRAINT uq_experiment_name UNIQUE (name)
);
GO

-- ─── model_runs ─────────────────────────────────────────────────────────────
IF OBJECT_ID('dbo.model_runs', 'U') IS NULL
CREATE TABLE dbo.model_runs (
    id               INT IDENTITY(1,1) PRIMARY KEY,
    experiment_id    INT           NOT NULL REFERENCES dbo.experiments(id),
    mlflow_run_id    NVARCHAR(64)  NULL,
    model_type       NVARCHAR(16)  NOT NULL DEFAULT 'MTL',  -- MTL | STL
    epoch            INT           NOT NULL,
    val_rmse         FLOAT         NULL,
    val_mae          FLOAT         NULL,
    val_pearson_r    FLOAT         NULL,
    val_auc_pose     FLOAT         NULL,
    val_auc_select   FLOAT         NULL,
    created_at       DATETIME2     NOT NULL DEFAULT SYSDATETIME()
);
GO

CREATE INDEX ix_model_runs_experiment ON dbo.model_runs(experiment_id);
GO

-- ─── binding_predictions ────────────────────────────────────────────────────
IF OBJECT_ID('dbo.binding_predictions', 'U') IS NULL
CREATE TABLE dbo.binding_predictions (
    id                INT IDENTITY(1,1) PRIMARY KEY,
    run_id            INT            NOT NULL REFERENCES dbo.model_runs(id),
    smiles            NVARCHAR(2048) NOT NULL,
    pocket_pdb        NVARCHAR(16)   NULL,
    pred_pkd          FLOAT          NULL,
    pred_pose_prob    FLOAT          NULL,
    pred_select_prob  FLOAT          NULL,
    created_at        DATETIME2      NOT NULL DEFAULT SYSDATETIME()
);
GO

CREATE INDEX ix_bp_run ON dbo.binding_predictions(run_id);
CREATE INDEX ix_bp_pkd  ON dbo.binding_predictions(pred_pkd DESC);
GO

-- ─── rl_molecules ───────────────────────────────────────────────────────────
IF OBJECT_ID('dbo.rl_molecules', 'U') IS NULL
CREATE TABLE dbo.rl_molecules (
    id             INT IDENTITY(1,1) PRIMARY KEY,
    experiment_id  INT            NOT NULL REFERENCES dbo.experiments(id),
    step           INT            NOT NULL,
    smiles         NVARCHAR(2048) NOT NULL,
    reward         FLOAT          NOT NULL,
    r_affinity     FLOAT          NULL,
    r_qed          FLOAT          NULL,
    r_sa           FLOAT          NULL,
    r_mw           FLOAT          NULL,
    pred_pkd       FLOAT          NULL,
    created_at     DATETIME2      NOT NULL DEFAULT SYSDATETIME()
);
GO

CREATE INDEX ix_rl_experiment ON dbo.rl_molecules(experiment_id);
CREATE INDEX ix_rl_reward     ON dbo.rl_molecules(reward DESC);
CREATE INDEX ix_rl_step       ON dbo.rl_molecules(step);
GO

-- ─── vina_benchmarks ────────────────────────────────────────────────────────
IF OBJECT_ID('dbo.vina_benchmarks', 'U') IS NULL
CREATE TABLE dbo.vina_benchmarks (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    smiles        NVARCHAR(2048) NOT NULL,
    pocket_pdb    NVARCHAR(16)   NOT NULL,
    vina_score    FLOAT          NULL,
    gnn_pred_pkd  FLOAT          NULL,
    created_at    DATETIME2      NOT NULL DEFAULT SYSDATETIME()
);
GO

-- ─── seed experiments ───────────────────────────────────────────────────────
IF NOT EXISTS (SELECT 1 FROM dbo.experiments WHERE name = 'gnn_mtl_baseline')
    INSERT INTO dbo.experiments (name, config_json)
    VALUES (
        'gnn_mtl_baseline',
        N'{"model":"HeteroGNN","heads":["affinity","pose","selectivity"],"loss":"MTL_kendall","epochs":20,"lr":0.001,"hidden":128,"num_layers":4}'
    );

IF NOT EXISTS (SELECT 1 FROM dbo.experiments WHERE name = 'gnn_stl_ablation')
    INSERT INTO dbo.experiments (name, config_json)
    VALUES (
        'gnn_stl_ablation',
        N'{"model":"HeteroGNN","heads":["affinity"],"loss":"MSE","epochs":20,"lr":0.001,"hidden":128,"num_layers":4}'
    );

IF NOT EXISTS (SELECT 1 FROM dbo.experiments WHERE name = 'rl_reinforce_egfr')
    INSERT INTO dbo.experiments (name, config_json)
    VALUES (
        'rl_reinforce_egfr',
        N'{"policy":"LSTM","hidden":512,"layers":2,"steps":300,"batch":32,"kl_beta":0.5,"reward_weights":{"affinity":0.5,"qed":0.2,"sa":0.2,"mw":0.1}}'
    );
GO

-- ─── seed Phase 2 training results ─────────────────────────────────────────
DECLARE @mtl_exp INT = (SELECT id FROM dbo.experiments WHERE name = 'gnn_mtl_baseline');
DECLARE @stl_exp INT = (SELECT id FROM dbo.experiments WHERE name = 'gnn_stl_ablation');

IF NOT EXISTS (SELECT 1 FROM dbo.model_runs WHERE experiment_id = @mtl_exp)
BEGIN
    INSERT INTO dbo.model_runs (experiment_id, model_type, epoch, val_rmse, val_mae, val_pearson_r, val_auc_pose, val_auc_select)
    VALUES
        (@mtl_exp, 'MTL', 19, 1.924, 1.551, 0.541, 0.778, 0.500),
        (@stl_exp, 'STL', 9,  2.034, 1.643, 0.489, NULL,  NULL);
END
GO

-- ─── seed RL molecules from Phase 3 results ─────────────────────────────────
DECLARE @rl_exp INT = (SELECT id FROM dbo.experiments WHERE name = 'rl_reinforce_egfr');

IF NOT EXISTS (SELECT 1 FROM dbo.rl_molecules WHERE experiment_id = @rl_exp)
BEGIN
    INSERT INTO dbo.rl_molecules (experiment_id, step, smiles, reward, r_affinity, r_qed, r_sa, r_mw, pred_pkd)
    VALUES
        (@rl_exp, 1,   'NS(=O)(=O)c1ccc(C(=O)N2CCC(O)(c3ccccc3)CC2)cc1', 0.709, 0.544, 0.865, 0.796, 1.0, 7.545),
        (@rl_exp, 2,   'NS(=O)(=O)c1ccc(NS(=O)(=O)c2ccc(Br)cc2)cc1',     0.703, 0.535, 0.830, 0.822, 1.0, 7.453),
        (@rl_exp, 3,   'O=C(O)Cc1cccc2ccccc12',                            0.700, 0.537, 0.782, 0.852, 1.0, 7.469),
        (@rl_exp, 4,   'O=S(=O)(Nc1nc2c(Cl)cc2s1)c1ccc(Cl)cc1F',         0.691, 0.521, 0.710, 0.823, 1.0, 7.328),
        (@rl_exp, 5,   'Cc1ccc(S(N)(=O)=O)cc1NC(=O)c1ccco1',             0.688, 0.518, 0.784, 0.800, 1.0, 7.298);
END
GO

PRINT 'GNNBindOptimizer schema initialised successfully.';
GO
