import warnings

import control as ct
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp


TOL = 1e-7


st.set_page_config(
    page_title="LGR - Sistema de Controle",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
    --bg: #f5f3ee;
    --ink: #1b1e23;
    --muted: #68707c;
    --panel: rgba(255, 255, 255, 0.86);
    --line: #d8d2c7;
    --accent: #0f766e;
    --accent-2: #b45309;
}

.stApp {
    background:
        radial-gradient(circle at 12% 18%, rgba(15, 118, 110, 0.16), transparent 28rem),
        radial-gradient(circle at 88% 6%, rgba(180, 83, 9, 0.12), transparent 24rem),
        linear-gradient(135deg, #f8f7f3 0%, #ebe6dc 52%, #f6f2ea 100%);
    color: var(--ink);
}

#MainMenu, footer { visibility: hidden; }

.block-container {
    padding-top: 2.1rem;
    padding-bottom: 3rem;
    max-width: 1260px;
}

[data-testid="stSidebar"] {
    background: #111827;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #f9fafb;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 0.38rem 0.55rem;
    margin-bottom: 0.25rem;
    background: rgba(255,255,255,0.045);
}

h1, h2, h3 {
    color: var(--ink);
    letter-spacing: 0;
}

.hero {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1.25rem 1.35rem;
    background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(250,247,240,0.80));
    box-shadow: 0 18px 42px rgba(31, 41, 55, 0.10);
}

.hero-title {
    font-size: 2.2rem;
    line-height: 1.1;
    font-weight: 760;
    margin: 0;
}

.hero-subtitle {
    margin-top: 0.55rem;
    color: var(--muted);
    max-width: 820px;
}

.step-shell {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--panel);
    padding: 1rem 1.1rem;
    margin-top: 0.8rem;
    box-shadow: 0 10px 24px rgba(31, 41, 55, 0.08);
}

.step-label {
    color: var(--accent);
    font-size: 0.8rem;
    font-weight: 760;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}

.callout {
    border-left: 4px solid var(--accent);
    padding: 0.65rem 0.9rem;
    background: rgba(15, 118, 110, 0.08);
    border-radius: 6px;
    margin: 0.75rem 0;
}

.callout.warning {
    border-left-color: var(--accent-2);
    background: rgba(180, 83, 9, 0.09);
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.72);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.75rem 0.85rem;
}

.block-container [data-testid="stWidgetLabel"] p,
.block-container [data-testid="stMetricLabel"] p,
.block-container [data-testid="stMetricValue"] {
    color: var(--ink) !important;
}

.block-container [data-testid="stMetric"] {
    min-height: 6.2rem;
}

.routh-wrap {
    margin: 0.9rem 0 1.2rem;
    overflow-x: auto;
}

.routh-table {
    border-collapse: collapse;
    min-width: 520px;
    max-width: 760px;
    background: rgba(255, 253, 248, 0.92);
    color: var(--ink);
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 10px 22px rgba(31, 41, 55, 0.08);
}

.routh-table th,
.routh-table td {
    border: 1px solid #d8d2c7;
    padding: 0.68rem 0.85rem;
    text-align: center;
    font-size: 1rem;
}

.routh-table th {
    background: #111827;
    color: #f9fafb;
    font-weight: 700;
}

.routh-table td:first-child {
    background: rgba(15, 118, 110, 0.10);
    color: #0f766e;
    font-weight: 760;
    width: 5.5rem;
}

.routh-table tr:nth-child(even) td:not(:first-child) {
    background: rgba(245, 243, 238, 0.78);
}
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def clean_coeffs(coeffs):
    arr = np.array(coeffs, dtype=float)
    if arr.size == 0:
        raise ValueError("Polinômio vazio.")
    first_nonzero = 0
    while first_nonzero < arr.size - 1 and abs(arr[first_nonzero]) < TOL:
        first_nonzero += 1
    return arr[first_nonzero:]


def parse_coeffs(text, label):
    try:
        coeffs = [float(part.replace(",", ".")) for part in text.split()]
    except ValueError as exc:
        raise ValueError(f"Use apenas números em {label}, separados por espaço.") from exc
    if not coeffs:
        raise ValueError(f"Informe ao menos um coeficiente em {label}.")
    return clean_coeffs(coeffs)


def fmt_number(value, digits=4):
    value = float(np.real_if_close(value))
    if abs(value) < 5e-10:
        value = 0.0
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text if text not in {"-0", ""} else "0"


def fmt_complex(value, digits=4):
    z = complex(np.real_if_close(value))
    real = 0.0 if abs(z.real) < 5e-10 else z.real
    imag = 0.0 if abs(z.imag) < 5e-10 else z.imag
    if abs(imag) < TOL:
        return fmt_number(real, digits)
    if abs(real) < TOL:
        return f"{fmt_number(imag, digits)}i"
    sign = "+" if imag >= 0 else "-"
    return f"{fmt_number(real, digits)} {sign} {fmt_number(abs(imag), digits)}i"


def fmt_poly(coeffs, variable="s", digits=4):
    coeffs = clean_coeffs(coeffs)
    degree = len(coeffs) - 1
    terms = []
    for i, coeff in enumerate(coeffs):
        if abs(coeff) < TOL:
            continue
        power = degree - i
        sign = "-" if coeff < 0 else "+"
        abs_coeff = abs(coeff)
        if power == 0:
            body = fmt_number(abs_coeff, digits)
        elif power == 1:
            body = variable if abs(abs_coeff - 1.0) < TOL else f"{fmt_number(abs_coeff, digits)}{variable}"
        else:
            body = f"{variable}^{{{power}}}" if abs(abs_coeff - 1.0) < TOL else f"{fmt_number(abs_coeff, digits)}{variable}^{{{power}}}"
        terms.append((sign, body))
    if not terms:
        return "0"
    first_sign, first_body = terms[0]
    result = first_body if first_sign == "+" else f"-{first_body}"
    for sign, body in terms[1:]:
        result += f" {sign} {body}"
    return result


def fmt_roots(roots):
    if len(roots) == 0:
        return r"\varnothing"
    ordered = sorted(roots, key=lambda z: (round(z.real, 8), round(z.imag, 8)))
    return ", ".join(fmt_complex(root) for root in ordered)


def fmt_factor(root):
    z = complex(np.real_if_close(root))
    if abs(z) < TOL:
        return "s"
    if abs(z.imag) < TOL:
        sign = "+" if z.real < 0 else "-"
        return rf"(s {sign} {fmt_number(abs(z.real))})"
    sign_real = "+" if z.real < 0 else "-"
    sign_imag = "+" if z.imag < 0 else "-"
    return rf"(s {sign_real} {fmt_number(abs(z.real))} {sign_imag} {fmt_number(abs(z.imag))}i)"


def fmt_factorization(roots):
    if len(roots) == 0:
        return "1"
    return "".join(fmt_factor(root) for root in roots)


def pad_left(coeffs, size):
    coeffs = np.array(coeffs, dtype=float)
    if coeffs.size >= size:
        return coeffs
    return np.pad(coeffs, (size - coeffs.size, 0))


def fmt_characteristic(den, num):
    size = max(len(den), len(num))
    den_p = pad_left(den, size)
    num_p = pad_left(num, size)
    degree = size - 1
    terms = []
    for i, (d_coeff, n_coeff) in enumerate(zip(den_p, num_p)):
        power = degree - i
        pieces = []
        if abs(d_coeff) >= TOL:
            pieces.append(fmt_number(d_coeff))
        if abs(n_coeff) >= TOL:
            if abs(n_coeff - 1.0) < TOL:
                k_piece = "K"
            elif abs(n_coeff + 1.0) < TOL:
                k_piece = "-K"
            else:
                k_piece = rf"{fmt_number(n_coeff)}K"
            pieces.append(k_piece)
        if not pieces:
            continue
        coeff_text = " + ".join(pieces).replace("+ -", "- ")
        if power == 0:
            term = coeff_text
        elif power == 1:
            term = f"({coeff_text})s" if len(pieces) > 1 else f"{coeff_text}s"
        else:
            term = f"({coeff_text})s^{{{power}}}" if len(pieces) > 1 else f"{coeff_text}s^{{{power}}}"
        terms.append(term)
    return " + ".join(terms).replace("+ -", "- ") + " = 0"


def real_roots_only(poles, zeros):
    roots = []
    for pole in poles:
        if abs(pole.imag) < TOL:
            roots.append(float(pole.real))
    for zero in zeros:
        if abs(zero.imag) < TOL:
            roots.append(float(zero.real))
    return roots


def real_axis_intervals(poles, zeros):
    roots = real_roots_only(poles, zeros)
    if not roots:
        return []
    unique = sorted(set(round(root, 10) for root in roots), reverse=True)
    intervals = []
    for i, start in enumerate(unique):
        if i + 1 < len(unique):
            end = unique[i + 1]
            mid = (start + end) / 2
            label = f"{fmt_number(start)} até {fmt_number(end)}"
        else:
            end = None
            mid = start - 1.0
            label = f"{fmt_number(start)} até -\\infty"
        count_right = sum(1 for root in roots if root > mid + TOL)
        intervals.append(
            {
                "start": start,
                "end": end,
                "mid": mid,
                "count_right": count_right,
                "is_lgr": count_right % 2 == 1,
                "label": label,
            }
        )
    return intervals


def is_on_real_lgr(x, poles, zeros):
    roots = real_roots_only(poles, zeros)
    if not roots:
        return False
    if any(abs(x - root) < 1e-5 for root in roots):
        return True
    return sum(1 for root in roots if root > x + TOL) % 2 == 1


def k_of_s(s, den, num):
    numerator = -np.polyval(den, s)
    denominator = np.polyval(num, s)
    if abs(denominator) < TOL:
        return np.nan
    return numerator / denominator


def breakaway_candidates(den, num, poles, zeros):
    d_den = np.polyder(den)
    d_num = np.polyder(num)
    equation = np.polysub(np.polymul(d_den, num), np.polymul(den, d_num))
    equation = clean_coeffs(equation)
    roots = np.roots(equation) if len(equation) > 1 else np.array([])
    candidates = []
    for root in roots:
        if abs(root.imag) < 1e-6:
            x = float(root.real)
            gain = k_of_s(x, den, num)
            if np.isfinite(gain):
                candidates.append(
                    {
                        "s": x,
                        "K": float(np.real(gain)),
                        "valid": float(np.real(gain)) > 0 and is_on_real_lgr(x, poles, zeros),
                    }
                )
    candidates.sort(key=lambda item: item["s"])
    return equation, candidates


def safe_margin(sys):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = ct.margin(sys)
        gm = float(values[0])
        wg = float(values[2])
        if np.isfinite(gm) and gm > 0 and np.isfinite(wg) and wg > 0:
            return gm, wg
    except Exception:
        pass
    return None, None

def angles_to_target(target, roots):
    rows = []
    for root in roots:
        if abs(target - root) > 1e-6:
            angle = np.degrees(np.angle(target - root))
            rows.append((root, angle))
    return rows


def normalize_180(angle):
    return ((angle + 180) % 360) - 180


def departure_angle(target, poles, zeros):
    zero_sum = sum(np.degrees(np.angle(target - zero)) for zero in zeros if abs(target - zero) > 1e-6)
    pole_sum = sum(np.degrees(np.angle(target - pole)) for pole in poles if abs(target - pole) > 1e-6)
    return (180 + zero_sum - pole_sum) % 360


def arrival_angle(target, poles, zeros):
    pole_sum = sum(np.degrees(np.angle(target - pole)) for pole in poles if abs(target - pole) > 1e-6)
    zero_sum = sum(np.degrees(np.angle(target - zero)) for zero in zeros if abs(target - zero) > 1e-6)
    return (180 + pole_sum - zero_sum) % 360


def point_angle(point, poles, zeros):
    zero_sum = sum(np.degrees(np.angle(point - zero)) for zero in zeros)
    pole_sum = sum(np.degrees(np.angle(point - pole)) for pole in poles)
    return zero_sum, pole_sum, normalize_180(pole_sum - zero_sum)


def point_gain(point, poles, zeros):
    prod_p = float(np.prod([abs(point - pole) for pole in poles])) if len(poles) else 1.0
    prod_z = float(np.prod([abs(point - zero) for zero in zeros])) if len(zeros) else 1.0
    if prod_z < TOL:
        return np.inf, prod_p, prod_z
    return prod_p / prod_z, prod_p, prod_z


def setup_axis(ax, xlim=None, ylim=None):
    ax.set_facecolor("#fffdf8")
    ax.axhline(0, color="#5f6671", linewidth=1)
    ax.axvline(0, color="#5f6671", linewidth=1)
    ax.grid(True, color="#d8d2c7", linestyle="--", linewidth=0.8, alpha=0.82)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.tick_params(colors="#333842")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


def plot_poles_zeros(ax, poles, zeros, label=True):
    if len(poles):
        ax.scatter(poles.real, poles.imag, marker="x", color="#b91c1c", s=96, linewidths=2.2, label="Polos" if label else None, zorder=4)
    if len(zeros):
        ax.scatter(zeros.real, zeros.imag, marker="o", facecolors="none", edgecolors="#0f766e", s=96, linewidths=2.2, label="Zeros" if label else None, zorder=4)


def default_limits(poles, zeros):
    points = list(poles) + list(zeros)
    if not points:
        return (-5, 5), (-5, 5)
    reals = [p.real for p in points]
    imags = [p.imag for p in points]
    min_x, max_x = min(reals), max(reals)
    max_y = max(abs(y) for y in imags + [0])
    span_x = max(4.0, max_x - min_x)
    span_y = max(4.0, 2 * max_y)
    return (min_x - 0.25 * span_x - 1, max_x + 0.25 * span_x + 1), (-span_y / 2 - 1, span_y / 2 + 1)


def status_line(text, kind="ok"):
    cls = "callout warning" if kind == "warning" else "callout"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def render_routh_table(rows):
    html_rows = []
    for row in rows:
        label = row[0]
        col1 = row[1] if len(row) > 1 else ""
        col2 = row[2] if len(row) > 2 else ""
        col3 = row[3] if len(row) > 3 else ""
        html_rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td>{col1}</td>"
            f"<td>{col2}</td>"
            f"<td>{col3}</td>"
            "</tr>"
        )
    html = (
        '<div class="routh-wrap">'
        '<table class="routh-table">'
        "<thead><tr><th>Linha</th><th>1a coluna</th><th>2a coluna</th><th>3a coluna</th></tr></thead>"
        f"<tbody>{''.join(html_rows)}</tbody>"
        "</table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def show_step(number, title, body_func):
    st.markdown(
        f"""
        <div class="step-shell">
            <div class="step-label">Passo {number}</div>
            <h3>{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    body_func()


with st.sidebar:
    st.markdown("### Roteiro")
    view = st.radio(
        "Escolha a etapa",
        [
            "Visão geral",
            "1. Polinômio característico",
            "2. Fatoração de P(s)",
            "3. Polos e zeros",
            "4. Eixo real",
            "5. Lugares separados",
            "6. Simetria",
            "7. Assíntotas",
            "8. Ponto de saída",
            "9. Routh-Hurwitz",
            "10. Ângulos de partida",
            "11. Condição de ângulo",
            "12. Cálculo de K",
            "13. Projeto de Controlador",
            "Gráfico completo",
            "Todos os passos",
        ],
        label_visibility="collapsed",
    )


st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Esboço do Lugar Geométrico das Raízes</div>
        <div class="hero-subtitle">
            Aplicação para montar o LGR seguindo a linha da apostila: sistema em malha aberta,
            memória de cálculo, gráficos intermediários e teste de pontos.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("### Definição do sistema")
input_col_1, input_col_2, input_col_3 = st.columns([1.1, 1.1, 0.8])
with input_col_1:
    st.markdown("**Planta sem o ganho K: G(s)**")
    num_g_input = st.text_input("Numerador de G(s)", "1")
    den_g_input = st.text_input("Denominador de G(s)", "1 8 32 0")
with input_col_2:
    st.markdown("**Realimentação: H(s)**")
    num_h_input = st.text_input("Numerador de H(s)", "1")
    den_h_input = st.text_input("Denominador de H(s)", "1 4")
with input_col_3:
    st.markdown("**Ponto de teste**")
    s_real = st.number_input("Parte real de s_i", value=-2.0, step=0.25, format="%.4f")
    s_imag = st.number_input("Parte imaginária de s_i", value=3.0, step=0.25, format="%.4f")


try:
    num_g = parse_coeffs(num_g_input, "Numerador de G(s)")
    den_g = parse_coeffs(den_g_input, "Denominador de G(s)")
    num_h = parse_coeffs(num_h_input, "Numerador de H(s)")
    den_h = parse_coeffs(den_h_input, "Denominador de H(s)")
    num_ol = clean_coeffs(np.polymul(num_g, num_h))
    den_ol = clean_coeffs(np.polymul(den_g, den_h))
    sys = ct.TransferFunction(num_ol, den_ol)
    zeros = np.roots(num_ol) if len(num_ol) > 1 or abs(num_ol[0]) > TOL else np.array([])
    poles = np.roots(den_ol)
    nz = len(zeros)
    npoles = len(poles)
    s_i = complex(s_real, s_imag)
except Exception as exc:
    st.error(str(exc))
    st.stop()


summary_1, summary_2, summary_3, summary_4 = st.columns(4)
summary_1.metric("Polos", npoles)
summary_2.metric("Zeros", nz)
summary_3.metric("Assíntotas", max(npoles - nz, 0))
summary_4.metric("Ramos do LGR", npoles)


def step_1():
    st.markdown("O ganho ajustável multiplica a malha aberta. Primeiro formamos:")
    st.latex(rf"P(s) = G(s)H(s) = \frac{{{fmt_poly(num_g)}}}{{{fmt_poly(den_g)}}} \cdot \frac{{{fmt_poly(num_h)}}}{{{fmt_poly(den_h)}}}")
    st.latex(rf"P(s) = \frac{{{fmt_poly(num_ol)}}}{{{fmt_poly(den_ol)}}}")
    st.markdown("A equação característica é:")
    st.latex(rf"1 + KP(s) = 0")
    st.latex(rf"1 + K\frac{{{fmt_poly(num_ol)}}}{{{fmt_poly(den_ol)}}} = 0")
    st.markdown("Multiplicando pelo denominador de P(s):")
    st.latex(rf"{fmt_characteristic(den_ol, num_ol)}")


def step_2():
    st.markdown("Os polos vêm das raízes de D(s), e os zeros vêm das raízes de N(s).")
    st.latex(rf"N(s) = {fmt_poly(num_ol)}")
    st.latex(rf"D(s) = {fmt_poly(den_ol)}")
    st.latex(rf"z = \{{{fmt_roots(zeros)}\}}")
    st.latex(rf"p = \{{{fmt_roots(poles)}\}}")
    st.markdown("A forma fatorada de malha aberta fica:")
    st.latex(rf"P(s) = \frac{{{fmt_factorization(zeros)}}}{{{fmt_factorization(poles)}}}")
    st.markdown(f"Logo, `n_p = {npoles}` e `n_z = {nz}`.")


def step_3():
    st.markdown("Marcamos `X` nos polos e `O` nos zeros. O LGR começa nos polos quando `K = 0`.")
    xlim, ylim = default_limits(poles, zeros)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    setup_axis(ax, xlim, ylim)
    plot_poles_zeros(ax, poles, zeros)
    ax.set_title("Polos e zeros de malha aberta")
    ax.legend(frameon=True)
    st.pyplot(fig)


def step_4():
    st.markdown("Um trecho do eixo real pertence ao LGR quando há número ímpar de polos e zeros reais à direita do trecho.")
    intervals = real_axis_intervals(poles, zeros)
    if not intervals:
        status_line("Este passo não gera trecho no eixo real porque não há polos ou zeros reais.", "warning")
        return
    for item in intervals:
        parity = "ímpar" if item["is_lgr"] else "par"
        result = "pertence ao LGR" if item["is_lgr"] else "não pertence ao LGR"
        st.latex(rf"\text{{Intervalo }} {item['label']}: {item['count_right']} \text{{ polos/zeros reais à direita }} ({parity}) \Rightarrow {result}")

    roots = real_roots_only(poles, zeros)
    min_x = min(roots) - 3
    max_x = max(roots) + 2
    fig, ax = plt.subplots(figsize=(8, 2.8))
    setup_axis(ax, (min_x, max_x), (-1.2, 1.2))
    ax.set_yticks([])
    for item in intervals:
        if item["is_lgr"]:
            start = item["start"]
            end = item["end"] if item["end"] is not None else min_x
            ax.plot([start, end], [0, 0], color="#b45309", linewidth=5, solid_capstyle="round", zorder=2)
    plot_poles_zeros(ax, poles, zeros, label=False)
    ax.set_title("Trechos válidos no eixo real")
    st.pyplot(fig)


def step_5():
    st.markdown("O número de lugares separados é o número de ramos que saem dos polos de malha aberta.")
    st.latex(rf"LS = n_p = {npoles}")
    if nz > npoles:
        status_line("O sistema tem mais zeros que polos; para LGR clássico, normalmente se trabalha com função própria.", "warning")


def step_6():
    st.markdown("Como os coeficientes do polinômio característico são reais, as raízes complexas aparecem em pares conjugados.")
    st.latex(r"\text{Se } s = \sigma + j\omega \text{ pertence ao LGR, então } \sigma - j\omega \text{ também pertence.}")
    status_line("O LGR é simétrico em relação ao eixo real.")


def step_7():
    if npoles <= nz:
        status_line(r"Este passo não é necessário porque não há zeros infinitos: \(n_p \leq n_z\).", "warning")
        return
    count = npoles - nz
    sigma_a = (sum(p.real for p in poles) - sum(z.real for z in zeros)) / count
    poles_sum = " + ".join(f"({fmt_number(p.real)})" for p in poles)
    zeros_sum = " + ".join(f"({fmt_number(z.real)})" for z in zeros) if nz else "0"
    st.markdown("As assíntotas aparecem porque alguns ramos terminam em zeros no infinito.")
    st.latex(rf"n_p - n_z = {npoles} - {nz} = {count}")
    st.latex(rf"\sigma_A = \frac{{\sum p_j - \sum z_i}}{{n_p - n_z}} = \frac{{{poles_sum} - ({zeros_sum})}}{{{count}}} = {fmt_number(sigma_a)}")
    angles = [(2 * q + 1) * 180 / count for q in range(count)]
    for q, angle in enumerate(angles):
        st.latex(rf"\phi_A(q={q}) = \frac{{(2q+1)180^\circ}}{{{count}}} = {fmt_number(angle)}^\circ")

    xlim, ylim = default_limits(poles, zeros)
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0], 8)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    setup_axis(ax, (min(xlim[0], sigma_a - span * 0.7), max(xlim[1], sigma_a + span * 0.7)), (-span * 0.55, span * 0.55))
    plot_poles_zeros(ax, poles, zeros)
    ax.scatter([sigma_a], [0], marker="s", color="#b45309", s=80, label="Centroide")
    for angle in angles:
        theta = np.radians(angle)
        ax.plot([sigma_a, sigma_a + span * np.cos(theta)], [0, span * np.sin(theta)], color="#334155", linestyle="--", linewidth=1.4)
        ax.text(sigma_a + (span * 0.42) * np.cos(theta), (span * 0.42) * np.sin(theta), f"{fmt_number(angle)}", color="#111827")
    ax.set_title("Assíntotas do LGR")
    ax.legend(frameon=True)
    st.pyplot(fig)


def step_8():
    equation, candidates = breakaway_candidates(den_ol, num_ol, poles, zeros)
    display_equation = -equation
    st.markdown("No eixo real, o ponto de saída ou entrada vem de `K(s)` e de sua derivada.")
    st.latex(rf"K(s) = -\frac{{D(s)}}{{N(s)}} = -\frac{{{fmt_poly(den_ol)}}}{{{fmt_poly(num_ol)}}}")
    st.latex(rf"\frac{{dK}}{{ds}} = 0 \Rightarrow {fmt_poly(display_equation)} = 0")
    if not candidates:
        status_line("A derivada não encontrou candidatos reais para ponto de saída ou chegada.", "warning")
        return
    valid = [item for item in candidates if item["valid"]]
    for item in candidates:
        verdict = "válido" if item["valid"] else "descartado"
        st.latex(rf"s = {fmt_number(item['s'])},\quad K = {fmt_number(item['K'])}\quad \Rightarrow\quad {verdict}")
    if not valid:
        status_line("Há candidatos reais, mas nenhum está em trecho de LGR com K positivo.", "warning")
        return

    roots = real_roots_only(poles, zeros)
    min_x = min(min(roots) - 1.5, min(item["s"] for item in valid) - 1.5)
    max_x = max(max(roots) + 1.0, max(item["s"] for item in valid) + 1.5)
    x_values = np.linspace(min_x, max_x, 500)
    y_values = np.array([np.real(k_of_s(x, den_ol, num_ol)) for x in x_values])
    finite_abs = np.abs(y_values[np.isfinite(y_values)])
    if finite_abs.size:
        cap = np.nanpercentile(finite_abs, 96)
        y_values[np.abs(y_values) > cap] = np.nan
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.set_facecolor("#fffdf8")
    ax.plot(x_values, y_values, color="#b91c1c", linewidth=2)
    for item in valid:
        ax.scatter([item["s"]], [item["K"]], color="#0f766e", s=70, zorder=4)
        ax.annotate(f"({fmt_number(item['s'])}; {fmt_number(item['K'])})", (item["s"], item["K"]), xytext=(8, 9), textcoords="offset points")
    ax.axhline(0, color="#5f6671", linewidth=1)
    ax.grid(True, color="#d8d2c7", linestyle="--")
    ax.set_xlabel("s")
    ax.set_ylabel("K(s)")
    ax.set_title("Curva K(s) para localizar saída/entrada")
    st.pyplot(fig)
    

def generate_symbolic_routh(den, num):
    K = sp.Symbol('K', real=True)
    s = sp.Symbol('s')
    
    # Cria os polinômios a partir das listas de coeficientes
    D = sum(float(c) * s**(len(den)-1-i) for i, c in enumerate(den))
    N = sum(float(c) * s**(len(num)-1-i) for i, c in enumerate(num))
    
    # Coeficientes da equação D(s) + K*N(s) = 0
    char_poly = sp.Poly(D + K * N, s)
    coeffs = char_poly.all_coeffs()
    deg = len(coeffs) - 1
    
    # Prepara a matriz
    cols = (deg + 2) // 2
    table = [[sp.S(0)] * cols for _ in range(deg + 1)]
    
    # Preenche as linhas s^n e s^(n-1), transformando 4.0 em 4 (limpeza)
    for i, c in enumerate(coeffs[0::2]):
        table[0][i] = sp.nsimplify(c, tolerance=1e-5) 
    for i, c in enumerate(coeffs[1::2]):
        table[1][i] = sp.nsimplify(c, tolerance=1e-5)
        
    # Calcula as linhas inferiores algebricamente
    for i in range(2, deg + 1):
        for j in range(cols - 1):
            pivot = table[i-1][0]
            if pivot == 0:
                pivot = sp.Symbol(r'\epsilon') # Evita divisão por zero
            
            # Determinante de Routh
            val = (pivot * table[i-2][j+1] - table[i-2][0] * table[i-1][j+1]) / pivot
            table[i][j] = sp.cancel(sp.simplify(val)) # Simplifica a expressão com K
            
    return table, deg


def step_9():
    st.markdown("O cruzamento do eixo imaginário ocorre quando o polinômio característico fica marginalmente estável.")
    st.latex(rf"{fmt_characteristic(den_ol, num_ol)}")
    
    # Gera e exibe a Tabela Simbólica usando o ambiente nativo array do LaTeX
    try:
        table, deg = generate_symbolic_routh(den_ol, num_ol)
        
        latex_str = r"\begin{array}{c|ccc}" + "\n"
        latex_str += r"\text{Linha} & \text{1a coluna} & \text{2a coluna} & \text{3a coluna} \\ \hline" + "\n"
        
        for i in range(deg + 1):
            label = f"s^{{{deg - i}}}"
            cols = []
            for val in table[i]:
                if val == 0:
                    cols.append("0")
                else:
                    cols.append(sp.latex(val))
            
            # Preenche com espaços vazios para garantir sempre 3 colunas de cálculo na exibição
            cols = cols[:3] + [""] * (3 - len(cols[:3]))
            row_str = f"{label} & {cols[0]} & {cols[1]} & {cols[2]} \\\\"
            latex_str += row_str + "\n"
            
        latex_str += r"\end{array}"
        
        st.markdown("**Tabela de Routh-Hurwitz:**")
        st.latex(latex_str)
        
    except Exception as e:
        status_line(f"Erro ao gerar tabela simbólica: {e}", "warning")
        
    # Usa a biblioteca control para achar o valor numérico do cruzamento
    gm, wg = safe_margin(sys)
    if gm is not None:
        st.markdown("Avaliando o ganho limite (onde a linha de $s^1$ ou $s^0$ zera):")
        st.latex(rf"K_{{marg}} = {fmt_number(gm)},\qquad s_{{1,2}} = \pm {fmt_number(wg)}i")
    else:
        status_line("Não foi encontrado cruzamento positivo do eixo imaginário para este sistema.", "warning")

def step_10():
    complex_poles = [pole for pole in poles if pole.imag > TOL]
    complex_zeros = [zero for zero in zeros if zero.imag > TOL]
    if not complex_poles and not complex_zeros:
        status_line("Este passo não é necessário porque não há polos ou zeros complexos no semiplano superior.", "warning")
        return
    st.latex(r"\angle P(s) = 180^\circ \pm q360^\circ")
    for index, pole in enumerate(complex_poles, start=1):
        parts = angles_to_target(pole, poles)
        zero_parts = angles_to_target(pole, zeros)
        pole_text = " + ".join(fmt_number(angle) for _, angle in parts) or "0"
        zero_text = " + ".join(fmt_number(angle) for _, angle in zero_parts) or "0"
        theta = departure_angle(pole, poles, zeros)
        st.markdown(f"Ângulo de partida no polo `p{index} = {fmt_complex(pole)}`:")
        st.latex(rf"\theta = 180^\circ + ({zero_text}) - ({pole_text}) = {fmt_number(theta)}^\circ")

    for index, zero in enumerate(complex_zeros, start=1):
        theta = arrival_angle(zero, poles, zeros)
        st.markdown(f"Ângulo de chegada no zero `z{index} = {fmt_complex(zero)}`:")
        st.latex(rf"\theta = {fmt_number(theta)}^\circ")

    target = complex_poles[0] if complex_poles else complex_zeros[0]
    xlim, ylim = default_limits(poles, zeros)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    setup_axis(ax, xlim, ylim)
    plot_poles_zeros(ax, poles, zeros)
    for root in list(poles) + list(zeros):
        if abs(root - target) > 1e-6:
            ax.annotate("", xy=(target.real, target.imag), xytext=(root.real, root.imag), arrowprops={"arrowstyle": "->", "color": "#475569", "linestyle": "dotted"})
    ax.scatter([target.real], [target.imag], color="#b45309", s=60, zorder=5)
    ax.set_title("Vetores usados na condição de ângulo")
    st.pyplot(fig)


def step_11():
    zero_sum, pole_sum, angle = point_angle(s_i, poles, zeros)
    st.markdown("Um ponto pertence ao LGR se a fase de `P(s_i)` for equivalente a `180 graus` módulo `360 graus`.")
    st.latex(rf"s_i = {fmt_complex(s_i)}")
    st.latex(rf"\angle P(s_i) = \sum \angle(s_i-p_j) - \sum \angle(s_i-z_i)")
    st.latex(rf"\angle P(s_i) = {fmt_number(pole_sum)}^\circ - ({fmt_number(zero_sum)}^\circ) \equiv {fmt_number(angle)}^\circ")
    if np.isclose(abs(angle), 180, atol=2.0):
        status_line("O ponto satisfaz a condição de ângulo e pertence ao LGR.")
    else:
        status_line("O ponto não satisfaz a condição de ângulo para K positivo.", "warning")


def step_12():
    gain, prod_p, prod_z = point_gain(s_i, poles, zeros)
    _, _, angle = point_angle(s_i, poles, zeros)
    st.markdown("Quando o ponto satisfaz a condição de ângulo, o ganho vem da condição de módulo[cite: 1].")
    st.latex(r"|KP(s_i)| = 1")
    st.latex(rf"K = \frac{{1}}{{|P(s_i)|}} = \frac{{\prod |s_i-p_j|}}{{\prod |s_i-z_i|}}")
    st.latex(rf"K = \frac{{{fmt_number(prod_p)}}}{{{fmt_number(prod_z)}}} = {fmt_number(gain)}")
    if not np.isclose(abs(angle), 180, atol=2.0):
        status_line("Como o ponto não passou no passo 11, este K é apenas o valor de módulo, não confirma pertencimento ao LGR.", "warning")


def step_13():
    st.markdown("### Parâmetros de Desempenho")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ctrl_type = st.selectbox("Controlador", ["PD", "PI", "PID (Zeros reais e iguais)"])
    with col2:
        mp = st.number_input("Sobressinal Mp (%)", value=10.0, step=1.0)
    with col3:
        ts = st.number_input("Tempo de Acomodação (s)", value=4.0, step=0.1)
        crit = st.selectbox("Critério", ["5%", "2%"])
        
    use_custom = st.checkbox("Fornecer polo desejado (s_d) diretamente")
    if use_custom:
        c1, c2 = st.columns(2)
        with c1:
            sd_real = st.number_input("Parte Real", value=-4.0, step=0.1)
        with c2:
            sd_imag = st.number_input("Parte Imag (positiva)", value=4.0, step=0.1)
        sd = complex(sd_real, abs(sd_imag))
        st.latex(rf"s_d = {fmt_complex(sd)}")
    else:
        zeta = np.sqrt((np.log(mp/100)**2) / (np.pi**2 + (np.log(mp/100))**2))
        wn = 4 / (zeta * ts) if crit == "2%" else 3 / (zeta * ts)
        wd = wn * np.sqrt(1 - zeta**2)
        sd = -zeta * wn + 1j * wd
        st.markdown("Cálculo do polo dominante desejado:")
        st.latex(rf"\zeta = {fmt_number(zeta)},\quad \omega_n = {fmt_number(wn)} \text{{ rad/s}}")
        st.latex(rf"s_d = -\zeta\omega_n + j\omega_d = {fmt_complex(sd)}")
        
    zero_sum, pole_sum, angle_P = point_angle(sd, poles, zeros)
    
    st.markdown("### Deficiência Angular e Compensação")
    st.markdown("Avaliamos a planta original $P(s)$ no polo desejado:")
    st.latex(rf"\angle P(s_d) = {fmt_number(angle_P)}^\circ")
    
    angle_sd = np.degrees(np.angle(sd)) % 360
    
    if ctrl_type == "PD":
        theta_z = (angle_P - 180) % 360
        st.latex(rf"\theta_z = \angle P(s_d) - 180^\circ = {fmt_number(theta_z)}^\circ")
    elif ctrl_type == "PI":
        theta_z = (angle_P + angle_sd - 180) % 360
        st.markdown("Para o PI, incluímos o ângulo do polo na origem:")
        st.latex(rf"\theta_z = (\angle P(s_d) + \angle s_d - 180^\circ) = {fmt_number(theta_z)}^\circ")
    else: 
        total_z = (angle_P + angle_sd - 180) % 360
        theta_z = total_z / 2
        st.markdown("Para o PID com zeros duplos, cada zero contribui com metade do necessário:")
        st.latex(rf"\theta_z = \frac{{\angle P(s_d) + \angle s_d - 180^\circ}}{{2}} = {fmt_number(theta_z)}^\circ")
        
    if theta_z <= 0 or theta_z >= 180:
        status_line(f"O ângulo necessário para o zero ({fmt_number(theta_z)}°) não está no intervalo (0, 180°), indicando que não pode ser realizado com um zero real à esquerda.", "warning")
        return
        
    z = (sd.imag / np.tan(np.radians(theta_z))) - sd.real
    st.latex(rf"\text{{Posição do zero no eixo real: }} z = \frac{{\omega_d}}{{\tan(\theta_z)}} - \sigma_d = {fmt_number(z)}")
    
    if ctrl_type == "PD":
        num_c = [1, z]
        den_c = [1]
        st.latex(rf"G_c(s) = K_c(s + {fmt_number(z)})")
    elif ctrl_type == "PI":
        num_c = [1, z]
        den_c = [1, 0]
        st.latex(rf"G_c(s) = K_c \frac{{s + {fmt_number(z)}}}{{s}}")
    else:
        num_c = clean_coeffs(np.polymul([1, z], [1, z]))
        den_c = [1, 0]
        st.latex(rf"G_c(s) = K_c \frac{{(s + {fmt_number(z)})^2}}{{s}}")
        
    new_num = clean_coeffs(np.polymul(num_ol, num_c))
    new_den = clean_coeffs(np.polymul(den_ol, den_c))
    new_zeros = np.roots(new_num) if len(new_num) > 1 else np.array([])
    new_poles = np.roots(new_den)
    
    gain, prod_p, prod_z = point_gain(sd, new_poles, new_zeros)
    st.markdown("Pela condição de módulo aplicada ao sistema já compensado:")
    st.latex(rf"K_c = \frac{{1}}{{|P_{{comp}}(s_d)|}} = {fmt_number(gain)}")
    
    st.markdown("### LGR do Sistema Compensado")
    sys_comp = ct.TransferFunction(new_num, new_den)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    setup_axis(ax)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rt, _ = ct.root_locus(sys_comp, plot=False)
    for branch in range(rt.shape[1]):
        ax.plot(rt[:, branch].real, rt[:, branch].imag, color="#2563eb", linewidth=2)
    plot_poles_zeros(ax, new_poles, new_zeros)
    
    ax.scatter([sd.real], [sd.imag], marker="*", color="#b91c1c", s=150, zorder=5, label="Polo Desejado (s_d)")
    ax.scatter([sd.real], [-sd.imag], marker="*", color="#b91c1c", s=150, zorder=5)
    
    xlim, ylim = default_limits(new_poles, new_zeros)
    all_re = rt.real[np.isfinite(rt.real)]
    all_im = rt.imag[np.isfinite(rt.imag)]
    if all_re.size and all_im.size:
        xlim = (min(xlim[0], np.nanpercentile(all_re, 2) - 1, sd.real - 1), max(xlim[1], np.nanpercentile(all_re, 98) + 1, sd.real + 1))
        ylim = (min(ylim[0], np.nanpercentile(all_im, 2) - 1, -sd.imag - 1), max(ylim[1], np.nanpercentile(all_im, 98) + 1, sd.imag + 1))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(f"Lugar Geométrico das Raízes (Com controlador {ctrl_type})")
    ax.legend(frameon=True)
    st.pyplot(fig)


def final_plot():
    st.markdown("O traçado completo é gerado variando `K` e calculando as raízes do polinômio característico.")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    setup_axis(ax)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        roots, _ = ct.root_locus(sys, plot=False)
    for branch in range(roots.shape[1]):
        ax.plot(roots[:, branch].real, roots[:, branch].imag, color="#2563eb", linewidth=2)
    plot_poles_zeros(ax, poles, zeros)
    xlim, ylim = default_limits(poles, zeros)
    all_re = roots.real[np.isfinite(roots.real)]
    all_im = roots.imag[np.isfinite(roots.imag)]
    if all_re.size and all_im.size:
        xlim = (min(xlim[0], np.nanpercentile(all_re, 2) - 1), max(xlim[1], np.nanpercentile(all_re, 98) + 1))
        ylim = (min(ylim[0], np.nanpercentile(all_im, 2) - 1), max(ylim[1], np.nanpercentile(all_im, 98) + 1))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("Lugar Geométrico das Raízes")
    ax.legend(frameon=True)
    st.pyplot(fig)


STEPS = {
    "1. Polinômio característico": ("Escrever o polinômio característico", step_1),
    "2. Fatoração de P(s)": ("Fatorar P(s) em polos e zeros", step_2),
    "3. Polos e zeros": ("Assinalar polos e zeros no plano s", step_3),
    "4. Eixo real": ("Assinalar os segmentos do eixo real", step_4),
    "5. Lugares separados": ("Determinar o número de lugares separados", step_5),
    "6. Simetria": ("Verificar simetria em relação ao eixo real", step_6),
    "7. Assíntotas": ("Calcular centroide e ângulos das assíntotas", step_7),
    "8. Ponto de saída": ("Determinar ponto de saída ou entrada", step_8),
    "9. Routh-Hurwitz": ("Determinar cruzamento do eixo imaginário", step_9),
    "10. Ângulos de partida": ("Determinar ângulos de partida e chegada", step_10),
    "11. Condição de ângulo": ("Testar se um ponto pertence ao LGR", step_11),
    "12. Cálculo de K": ("Calcular o ganho pelo módulo", step_12),
    "13. Projeto de Controlador": ("Projeto de Controlador (PD, PI, PID)", step_13),
}


def overview():
    st.markdown("### Visão geral")
    left, right = st.columns([1, 1])
    with left:
        st.latex(rf"G(s)=\frac{{{fmt_poly(num_g)}}}{{{fmt_poly(den_g)}}}")
        st.latex(rf"H(s)=\frac{{{fmt_poly(num_h)}}}{{{fmt_poly(den_h)}}}")
        st.latex(rf"P(s)=G(s)H(s)=\frac{{{fmt_poly(num_ol)}}}{{{fmt_poly(den_ol)}}}")
    with right:
        st.markdown("**Passos que dependem do sistema**")
        st.write(f"Assíntotas: {'necessário' if npoles > nz else 'não necessário'}")
        st.write(f"Ponto de saída/entrada: {'verificar candidatos' if real_axis_intervals(poles, zeros) else 'não necessário'}")
        st.write(f"Ângulos de partida/chegada: {'necessário' if any(p.imag > TOL for p in poles) or any(z.imag > TOL for z in zeros) else 'não necessário'}")
        gm, _ = safe_margin(sys)
        st.write(f"Cruzamento no eixo imaginário: {'existe' if gm else 'não encontrado para K positivo'}")
    final_plot()


if view == "Visão geral":
    overview()
elif view == "Todos os passos":
    for index, (_, (title, func)) in enumerate(STEPS.items(), start=1):
        show_step(index, title, func)
    show_step("Final", "Gráfico completo do LGR", final_plot)
elif view == "Gráfico completo":
    show_step("Final", "Gráfico completo do LGR", final_plot)
else:
    title, func = STEPS[view]
    number = view.split(".")[0]
    show_step(number, title, func)