# Architecture

Two views of the application: the layer/data-flow structure, and the generation
pipeline in run order. Both are Mermaid; GitHub and most editors render them inline.

## Layers and data flow

```mermaid
flowchart TB
    subgraph CLI["worldgen/cli.py — click commands"]
        G[generate]
        R[render]
        E[export]
        IH[import-heightmap]
        IC[init-config]
        P[presets]
    end

    subgraph CORE["core/ — types + orchestration, no I/O"]
        CFG[config.py<br/>WorldConfig · ClimateContext]
        PIPE[pipeline.py<br/>GeneratorStage · GeneratorPipeline]
        WS[world_state.py<br/>WorldState · River · Road · Ferry<br/>schema v1.2]
        HEX[hex.py<br/>Hex · TerrainClass · Settlement]
        GRID[hex_grid.py<br/>axial / offset layouts]
        ERR[errors.py]
    end

    subgraph STAGES["stages/ — pure WorldState -> WorldState"]
        REG[__init__.py<br/>default_stages / stages_for]
        SEQ[16 stage classes]
    end

    subgraph EXPORT["export/ — all file I/O"]
        JS[json_export]
        SVG[svg_export]
        PNG[png_export]
        LEG[legend]
        RIV[rivers]
        HI[heightmap_import<br/>HeightmapError]
    end

    subgraph RENDER["render/ — matplotlib debug viewer"]
        DV[debug_viewer.render]
    end

    G --> CFG
    G --> REG
    IH --> REG
    REG --> SEQ
    G --> PIPE
    IH --> PIPE
    PIPE -->|seeded child RNG per stage| SEQ
    SEQ -->|mutates| WS
    PIPE --> WS
    WS --- HEX
    WS --- GRID
    SEQ -.reads.-> CFG

    SEQ -->|ImageElevationStage| HI
    G --> JS
    G --> DV
    IH --> JS
    IH --> DV
    R --> JS
    R --> DV
    E --> JS
    E --> SVG
    SVG --- LEG
    PNG --- LEG
    SVG --- RIV
    PNG --- RIV

    style CORE fill:#eef4ff
    style STAGES fill:#eefaee
    style EXPORT fill:#fff4e6
    style RENDER fill:#f6eeff
```

## Generation pipeline (run order)

```mermaid
flowchart LR
    A[Elevation<br/><i>or ImageElevation<br/>if heightmap_path</i>] --> B[Erosion]
    B --> C[TerrainClassification]
    C --> D[WaterBodies]
    D --> E[Hydrology]
    E --> F[Climate]
    F --> G[Biome]
    G --> H[LandCover]
    H --> I[Habitability]
    I --> J[CityTown]
    J --> K[InterurbanRoad]
    K --> L[Cultivation]
    L --> M[VillagePlacement]
    M --> N[VillageTrack]
    N --> O[VillageCultivation]

    A -.-> S(["WorldState<br/>hexes · rivers · roads<br/>settlements · ferries"])
    O -.-> S
```

## Structural notes

- `stages_for` swaps `ImageElevationStage` into slot 0 positionally, so the tuple length
  never changes. `GeneratorPipeline.run` draws a child seed per stage from the parent
  stream, so an imported world sees exactly the seeds a generated one would.
- `import-heightmap` runs only slot 0 plus `TerrainClassificationStage` — no erosion.
  That omission is what makes it the faithful path: elevations are what the image says,
  where `generate` would renormalise them.
- The settlement block (CityTown → VillageCultivation) is order-load-bearing: roads and
  cultivation must exist before `VillagePlacementStage` will site villages.
- The stage list lives in `worldgen/stages/__init__.py` and nowhere else; the CLI and the
  tests both read it from there.
