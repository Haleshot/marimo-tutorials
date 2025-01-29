# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.9.32"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # An Interactive Primer of New Tricks used by DeepSeek's v3-based-R1

        Welcome to this exploration/write-up of DeepSeek-v3-based- DeepSeek-R1's architectural innovations! This notebook breaks down the key technical breakthroughs that have enabled DeepSeek to achieve performance on par with leading closed-source models.

        **Key Highlights:**

        - SOTA level performance (comparable to leading closed models)

        - Novel Multi-Head Latent Attention (MLA)

        - DeepseekMoE architecture with auxiliary-loss-free load balancing

        - Advanced training techniques including Multi-Token Prediction

        Let's dive into these innovations through interactive visualizations and examples!
        """
    ).center()
    return


@app.cell
def __(overview):
    overview
    return


@app.cell
def __(key_features):
    key_features
    return


@app.cell(hide_code=True)
def _(mo):
    overview = mo.md(
        r"""
        ## Model Overview

        DeepSeek-V3 is a powerful Mixture-of-Experts (MoE) language model with:

        - 671B total parameters

        - 37B activated parameters per token

        - Training on 14.8 trillion diverse tokens
        """
    )

    key_features = mo.callout(
        mo.md("""
            **Key Architectural Features:**

            1. Multi-Head Latent Attention (MLA)

            2. DeepSeekMoE architecture

            3. Auxiliary-loss-free load balancing

            4. Multi-token prediction objective
        """),
        kind="info"
    )
    return key_features, overview


@app.cell
def __(mla_intro):
    mla_intro
    return


@app.cell
def __(mha_basics):
    mha_basics
    return


@app.cell(hide_code=True)
def _(mo):
    mla_intro = mo.md(
        r"""
        ## 1. Multi-Head Latent Attention (MLA)

        Let's first understand the evolution of attention mechanisms and how MLA improves upon them.
        """
    )

    mha_basics = mo.md(
        r"""
        ### Basic Building Blocks

        For an input token $h_t$ with embedding dimension $d$:

        - $n_h$: number of attention heads
        - $d_h$: dimension per head

        The core transformations are:

        $$
        \begin{align*}
        q_t &= W_Q @ h_t \text{ (Query)} \\
        k_t &= W_K @ h_t \text{ (Key)} \\
        v_t &= W_V @ h_t \text{ (Value)}
        \end{align*}
        $$
        """
    )
    return mha_basics, mla_intro


@app.cell(hide_code=True)
def _(mo):
    # memory calculator
    n_heads = mo.ui.slider(
        start=1,
        stop=32,
        value=8,
        step=1,
        label="Number of Attention Heads (n_h)"
    )

    dim_per_head = mo.ui.slider(
        start=32,
        stop=128,
        value=64,
        step=16,
        label="Dimension per Head (d_h)"
    )

    seq_length = mo.ui.slider(
        start=128,
        stop=2048,
        value=512,
        step=128,
        label="Sequence Length (L)"
    )

    _controls = mo.vstack([
        mo.md("### Memory Requirements Explorer"),
        mo.hstack([n_heads, dim_per_head, seq_length])
    ])
    _controls
    return dim_per_head, n_heads, seq_length


@app.cell
def __(comparison):
    comparison
    return


@app.cell
def __(insights):
    insights
    return


@app.cell(hide_code=True)
def _(dim_per_head, mo, n_heads, seq_length):
    def calculate_memory():
        mha_memory = 2 * n_heads.value * dim_per_head.value * seq_length.value
        gqa_memory = 2 * (n_heads.value // 2) * dim_per_head.value * seq_length.value  # Assuming 2 groups
        mqa_memory = 2 * dim_per_head.value * seq_length.value
        mla_memory = (4.5 * dim_per_head.value) * seq_length.value  # Approximate MLA memory

        return {
            "MHA": mha_memory,
            "GQA": gqa_memory,
            "MQA": mqa_memory,
            "MLA": mla_memory
        }

    memories = calculate_memory()
    comparison = mo.md(
        f"""
        ### Memory Comparison (elements)

        - MHA: {memories['MHA']:,}
        - GQA: {memories['GQA']:,}
        - MQA: {memories['MQA']:,}
        - MLA: {memories['MLA']:,}
        """
    )

    insights = mo.callout(
        mo.md("""
            **Key Insights:**

            - MLA achieves memory efficiency close to MQA

            - Performance remains similar to MHA

            - Solves RoPE compatibility through decoupling
        """),
        kind="success"
    )
    return calculate_memory, comparison, insights, memories


@app.cell
def __(moe_intro):
    moe_intro
    return


@app.cell
def __(expert_types):
    expert_types
    return


@app.cell(hide_code=True)
def _(mo):
    moe_intro = mo.md(
        r"""
        ## 2. DeepSeekMoE Architecture

        DeepSeekMoE introduces a novel approach to Mixture of Experts with:

        - Finer-grained experts

        - Separation of shared and routed experts
        """
    )

    expert_types = mo.accordion({
        "Types of Experts": mo.md("""

            1. **Shared Experts**

            - Common resources used by all tokens

            - Always activated

            2. **Routed Experts**

            - Specialized for specific tasks

            - Activated based on input token characteristics
        """)

    })
    return expert_types, moe_intro


@app.cell
def __(math_details):
    math_details
    return


@app.cell(hide_code=True)
def _(mo):
    math_details = mo.md(
        r"""
        ### Mathematical Formulation

        The output of the FFN layer $h'_t$ is computed as:

        $$
        \mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \cdot \text{FFN}_i^{(r)}(\mathbf{u}_t)
        $$

        where:

        - $N_s$: Number of shared experts

        - $N_r$: Number of routed experts

        - $g_{i,t}$: Gating weight for routed expert $i$
        """
    )
    return (math_details,)


@app.cell(hide_code=True)
def _(mo):
    # Interactive MoE simulator
    n_shared = mo.ui.slider(
        start=1,
        stop=8,
        value=2,
        step=1,
        label="Number of Shared Experts"
    )

    n_routed = mo.ui.slider(
        start=2,
        stop=16,
        value=8,
        step=2,
        label="Number of Routed Experts"
    )

    k_active = mo.ui.slider(
        start=1,
        stop=4,
        value=2,
        step=1,
        label="Active Routed Experts (k)"
    )

    _controls = mo.vstack([
        mo.md("### MoE Parameter Explorer"),
        mo.hstack([n_shared, n_routed, k_active])
    ])
    _controls
    return k_active, n_routed, n_shared


@app.cell
def __(results_display):
    results_display
    return


@app.cell
def __(moe_insights):
    moe_insights
    return


@app.cell
def __(k_active, mo, n_routed, n_shared):
    def calculate_moe():
        # simplified approx. of total parms
        base_expert_size = 1024 * 4096  # for one expert (feed-forward layer)
        
        # Total parameters
        total_params = (n_shared.value + n_routed.value) * base_expert_size
        
        # Active parameters per forward pass
        active_params = (n_shared.value + k_active.value) * base_expert_size
        
        # Calculating efficiency metrics
        parameter_efficiency = (active_params / total_params) * 100
        
        # Calculate theoretical compute savings
        compute_savings = (1 - (active_params / total_params)) * 100
        
        # Calculate load balance score (simplified) but
        # assuming perfect distribution across routed experts
        theoretical_load = k_active.value / n_routed.value
        load_balance_score = min(theoretical_load * 100, 100)

        return {
            "total_parameters": total_params,
            "active_parameters": active_params,
            "parameter_efficiency": parameter_efficiency,
            "compute_savings": compute_savings,
            "load_balance_score": load_balance_score
        }

    stats = calculate_moe()

    results_display = mo.md(
        f"""
        ### MoE Architecture Statistics
        
        - Total Parameters: {stats['total_parameters']:,}
        - Active Parameters per Forward Pass: {stats['active_parameters']:,}
        - Parameter Efficiency: {stats['parameter_efficiency']:.2f}%
        - Compute Savings: {stats['compute_savings']:.2f}%
        - Load Balance Score: {stats['load_balance_score']:.2f}%
        """
    )

    moe_insights = mo.callout(
        mo.md(f"""
            **Key Insights:**
            
            - {k_active.value} active experts out of {n_routed.value} routed experts
            
            - {n_shared.value} shared experts always active
            
            - Achieving {stats['compute_savings']:.1f}% compute savings
        """),
        kind="success"
    )
    return calculate_moe, moe_insights, results_display, stats


@app.cell
def __(balancing_intro):
    balancing_intro
    return


@app.cell
def __(algorithm):
    algorithm
    return


@app.cell(hide_code=True)
def _(mo):
    balancing_intro = mo.md(
        r"""
        ## 3. Auxiliary-Loss-Free Load Balancing

        DeepSeek introduces a novel approach to load balancing that doesn't require auxiliary loss terms:

        1. Uses bias terms only for routing decisions
        2. Maintains original affinity scores for actual usage
        3. Dynamically adjusts based on expert utilization
        """
    )

    algorithm = mo.callout(
        mo.md("""
            **Load Balancing Algorithm:**
            ```python
            For each expert after training step:
                If expert is overloaded:
                    bias -= γ
                If expert is underloaded:
                    bias += γ
            ```
        """),
        kind="neutral"
    )
    return algorithm, balancing_intro


@app.cell
def __(grpo_intro):
    grpo_intro
    return


@app.cell
def __(components):
    components
    return


@app.cell(hide_code=True)
def _(mo):
    grpo_intro = mo.md(
        r"""
        ## 4. Group Relative Policy Optimization (GRPO)

        GRPO is a key innovation in DeepSeek-R1 that optimizes policy without ground truth:

        $$
        J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q),\{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} 
        \left[\frac{1}{G}\sum_{i=1}^G \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}A_i, 
        \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right)A_i\right)\right]
        $$
        """
    )

    components = mo.accordion({
        "Key Components": mo.md("""
            1. **Expectation Part**: Considers all possible scenarios
            2. **Min and Clip**: Ensures stable, incremental updates
            3. **KL Divergence**: Keeps policy close to reference
        """)
    })
    return components, grpo_intro


@app.cell
def __(conclusion):
    conclusion
    return


@app.cell(hide_code=True)
def _(mo):
    conclusion = mo.md(
        r"""
        ## Summary

        DeepSeek's innovations combine to create a powerful and efficient model:

        1. **MLA** reduces memory requirements while maintaining performance
        2. **DeepSeekMoE** provides efficient expert specialization
        3. **Auxiliary-Loss-Free Load Balancing** ensures optimal expert utilization
        4. **GRPO** enables effective policy optimization

        These techniques together enable DeepSeek to achieve performance comparable to leading closed-source models while maintaining efficiency and stability during training.
        """
    )
    return (conclusion,)


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
