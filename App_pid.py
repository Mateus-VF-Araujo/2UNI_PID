import warnings
import control as ct
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp


TOL = 1e-7


st.set_page_config(
    page_title="Projeto de Controladores",
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

.block-container div[role="radiogroup"] label p {
    color: #000000 !important;
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


def normalize_180(angle):
    return ((angle + 180) % 360) - 180


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
            "1. Polinômio característico",
            "2. Fatoração de P(s)",
            "3. Polos e zeros",
            "4. Projeto de Controlador",
            "5. Discretização",
            "Todos os passos",
        ],
        label_visibility="collapsed",
    )


st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Projeto de Controladores</div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("### Definição do sistema")
input_col_1, input_col_2 = st.columns([1, 1])
with input_col_1:
    st.markdown("**Planta sem o ganho K: G(s)**")
    num_g_input = st.text_input("Numerador de G(s)", "1")
    den_g_input = st.text_input("Denominador de G(s)", "1 8 32 0")
with input_col_2:
    st.markdown("**Realimentação: H(s)**")
    num_h_input = st.text_input("Numerador de H(s)", "1")
    den_h_input = st.text_input("Denominador de H(s)", "1 4")


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
except Exception as exc:
    st.error(str(exc))
    st.stop()


summary_1, summary_2, summary_3, summary_4 = st.columns(4)
summary_1.metric("Polos", npoles)
summary_2.metric("Zeros", nz)
summary_3.metric("Assíntotas", max(npoles - nz, 0))

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
    st.markdown("Marcamos `X` nos polos e `O` nos zeros. `K = 0`.")
    xlim, ylim = default_limits(poles, zeros)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    setup_axis(ax, xlim, ylim)
    plot_poles_zeros(ax, poles, zeros)
    ax.set_title("Polos e zeros de malha aberta")
    ax.legend(frameon=True)
    st.pyplot(fig)


def step_4():
    st.markdown("### Projeto de Controladores pelo Método do LGR")
    
    ctrl_type = st.selectbox("Controlador", ["PD", "PI", "PID"])
    
    st.markdown("#### 1) Definir especificações e localizar o polo desejado ($s_d$)")
    
    input_method = st.radio(
        "Método de definição das especificações de malha fechada:",
        ["Métricas de Desempenho (Mp e ts)", "Parâmetros de 2ª Ordem (ζ e ωn)", "Polo Desejado Direto (sd)"],
        horizontal=True
    )
    
    if input_method == "Métricas de Desempenho (Mp e ts)":
        col1, col2, col3 = st.columns(3)
        with col1:
            mp = st.number_input("Sobressinal Mp (%)", value=10.0, step=1.0)
        with col2:
            ts = st.number_input("Tempo de Acomodação (s)", value=4.0, step=0.1)
        with col3:
            crit = st.selectbox("Critério", ["5%", "2%"])
            
        zeta = np.sqrt((np.log(mp/100)**2) / (np.pi**2 + (np.log(mp/100))**2))
        wn_mult = 4 if crit == "2%" else 3
        wn = wn_mult / (zeta * ts)
        wd = wn * np.sqrt(1 - zeta**2)
        sd = -zeta * wn + 1j * wd
        
        st.markdown("A partir da equação do máximo pico e tempo de acomodação, obtemos:")
        st.latex(rf"M_p = e^{{-\frac{{\zeta\pi}}{{\sqrt{{1-\zeta^2}}}}}} = {mp}\% \Rightarrow \zeta_{{min}} = {fmt_number(zeta)}")
        st.latex(rf"t_s = \frac{{{wn_mult}}}{{\zeta\omega_n}} = {ts}\text{{s}} \Rightarrow (\zeta\omega_n)_{{min}} = {fmt_number(zeta*wn)}")
        st.markdown("Sendo $\omega_d = \omega_n\sqrt{1-\zeta^2}$, os polos dominantes de malha fechada devem estar em:")
        st.latex(rf"s_d = -\zeta\omega_n \pm j\omega_d = {fmt_complex(sd)}")

    elif input_method == "Parâmetros de 2ª Ordem (ζ e ωn)":
        col1, col2 = st.columns(2)
        with col1:
            zeta = st.number_input("Fator de Amortecimento (ζ)", value=0.7, step=0.05)
        with col2:
            wn = st.number_input("Frequência Natural (ωn) [rad/s]", value=0.5, step=0.1)
            
        wd = wn * np.sqrt(1 - zeta**2)
        sd = -zeta * wn + 1j * wd
        
        st.markdown("Com os parâmetros fornecidos diretamente:")
        st.latex(rf"\zeta = {fmt_number(zeta)} \quad \text{{e}} \quad \omega_n = {fmt_number(wn)} \text{{ rad/s}}")
        st.latex(rf"\omega_d = \omega_n\sqrt{{1-\zeta^2}} = {fmt_number(wd)} \text{{ rad/s}}")
        st.markdown("Os polos dominantes de malha fechada são calculados por $s_d = -\zeta\omega_n \pm j\omega_d$:")
        st.latex(rf"s_d = -({fmt_number(zeta)})({fmt_number(wn)}) + j({fmt_number(wd)}) = {fmt_complex(sd)}")

    else:
        c1, c2 = st.columns(2)
        with c1:
            sd_real = st.number_input("Parte Real", value=-1.0, step=0.1)
        with c2:
            sd_imag = st.number_input("Parte Imag (positiva)", value=1.0, step=0.1)
        sd = complex(sd_real, abs(sd_imag))
        st.markdown("Polo dominante de malha fechada definido manualmente:")
        st.latex(rf"s_d = {fmt_complex(sd)}")

    # 2. Configurar os polos base do controlador
    if ctrl_type == "PD":
        st.markdown("#### 2) Verificar se o objetivo não pode ser atingido com um controlador Proporcional")
        poles_c = []
    elif ctrl_type == "PI":
        st.markdown("#### 2) Determinar o zero de modo que a condição de ângulo seja satisfeita")
        poles_c = [0.0]
    else:
        st.markdown("#### 2) Localizar o polo na origem e os zeros de modo que a condição de ângulo seja satisfeita")
        poles_c = [0.0]

    # 3. Condição de Ângulo
    sys_poles = list(poles) + poles_c
    sys_zeros = list(zeros)
    
    st.markdown("#### 3) Determinar o zero utilizando a condição de ângulo")
    st.markdown("Cálculo geométrico dos ângulos desde as singularidades de malha aberta até $s_d$:")
    
    pole_angles = []
    for i, p in enumerate(sys_poles, 1):
        re_diff = sd.real - np.real(p)
        im_diff = sd.imag - np.imag(p)
        ang = np.degrees(np.angle(sd - p))
        pole_angles.append(ang)
        st.latex(rf"\theta_{{p{i}}} = \angle(s_d - p_{i}) = \text{{ATAN2}}({fmt_number(im_diff)}, {fmt_number(re_diff)}) = {fmt_number(ang)}^\circ")
        
    zero_angles = []
    for i, z in enumerate(sys_zeros, 1):
        re_diff = sd.real - np.real(z)
        im_diff = sd.imag - np.imag(z)
        ang = np.degrees(np.angle(sd - z))
        zero_angles.append(ang)
        st.latex(rf"\theta_{{z{i}}} = \angle(s_d - z_{i}) = \text{{ATAN2}}({fmt_number(im_diff)}, {fmt_number(re_diff)}) = {fmt_number(ang)}^\circ")

    sum_p_text = " + ".join([f"{fmt_number(a)}^\circ" for a in pole_angles]) if pole_angles else "0^\circ"
    sum_z_text = " + ".join([f"{fmt_number(a)}^\circ" for a in zero_angles]) if zero_angles else "0^\circ"
    angle_sys = sum(zero_angles) - sum(pole_angles)
    
    st.markdown("Avaliando o ângulo total da malha aberta original no ponto $s_d$:")
    st.latex(rf"\angle P(s_d) = \sum \theta_z - \sum \theta_p = ({sum_z_text}) - ({sum_p_text})")
    st.latex(rf"\angle P(s_d) = {fmt_number(angle_sys)}^\circ")
    
    theta_total = normalize_180(-180 - angle_sys)
    if theta_total < 0: theta_total += 360
    
    if ctrl_type in ["PD", "PI"]:
        phi = theta_total
        st.markdown("O ângulo a compensar pelo zero do controlador é:")
        st.latex(rf"\phi = 180^\circ - \text{{deficiência angular}} \Rightarrow \phi = {fmt_number(phi)}^\circ")
        if phi <= 0 or phi >= 180:
            status_line("Erro: A especificação angular inviabiliza a colocação geométrica do zero.", "warning")
            return
        z_val = abs(sd.real) + sd.imag / np.tan(np.radians(phi))
        st.latex(rf"\text{{Parâmetro }} z = \frac{{\omega_d}}{{\tan(\phi)}} + |\sigma_d| = {fmt_number(z_val)}")
        st.markdown(rf"Isso significa que a raiz geométrica do controlador fica no ponto $s = -z = -{fmt_number(z_val)}$.")
        zeros_c = [-z_val]
    else: 
        phi = theta_total / 2
        st.markdown("Como os zeros são idênticos, a contribuição divide-se por dois:")
        st.latex(rf"\phi = \frac{{{fmt_number(theta_total)}^\circ}}{{2}} = {fmt_number(phi)}^\circ")
        if phi <= 0 or phi >= 180:
            status_line("Erro: Impossível localizar o par de zeros.", "warning")
            return
        z_val = abs(sd.real) + sd.imag / np.tan(np.radians(phi))
        st.latex(rf"\text{{Parâmetro }} z = \frac{{\omega_d}}{{\tan(\phi)}} + |\sigma_d| = {fmt_number(z_val)}")
        st.markdown(rf"As raízes geométricas ficam em $s = -z = -{fmt_number(z_val)}$.")
        zeros_c = [-z_val, -z_val]

    # 4. Condição de Módulo com ajuste do ganho da planta
    st.markdown("#### 4) Calcular o ganho total requerido, aplicando a condição de módulo")
    st.markdown("Cálculo discriminado do módulo dos vetores (distâncias reais) até $s_d$:")
    
    all_poles = sys_poles
    all_zeros = sys_zeros + zeros_c
    
    p_dists = []
    for i, p in enumerate(all_poles, 1):
        d = abs(sd - p)
        p_dists.append(d)
        st.latex(rf"|s_d - p_{i}| = \sqrt{{ ({fmt_number(sd.real)} - ({fmt_number(np.real(p))}))^2 + ({fmt_number(sd.imag)} - ({fmt_number(np.imag(p))}))^2 }} = {fmt_number(d)}")

    z_dists = []
    for i, z in enumerate(all_zeros, 1):
        d = abs(sd - z)
        z_dists.append(d)
        st.latex(rf"|s_d - z_{i}| = \sqrt{{ ({fmt_number(sd.real)} - ({fmt_number(np.real(z))}))^2 + ({fmt_number(sd.imag)} - ({fmt_number(np.imag(z))}))^2 }} = {fmt_number(d)}")

    prod_p = np.prod(p_dists) if p_dists else 1.0
    prod_z = np.prod(z_dists) if z_dists else 1.0
    K_total = prod_p / prod_z
    
    # Extração automática do ganho intrínseco da malha (funciona para qualquer planta)
    K_planta = abs(float(num_ol[0]) / float(den_ol[0]))
    Kc = K_total / K_planta
    
    str_num = " \cdot ".join([fmt_number(d) for d in p_dists]) if p_dists else "1"
    str_den = " \cdot ".join([fmt_number(d) for d in z_dists]) if z_dists else "1"

    st.markdown("Aplicando o balanço de distâncias na equação geral:")
    st.latex(rf"K_{{LGR}} = \frac{{\prod |s_d - p_j|}}{{\prod |s_d - z_i|}} = \frac{{{str_num}}}{{{str_den}}} = {fmt_number(K_total)}")
    st.markdown("Como a malha aberta possui um ganho multiplicador inerente, isolamos o ganho do controlador:")
    st.latex(rf"K_c = \frac{{K_{{LGR}}}}{{K_{{planta}}}} = \frac{{{fmt_number(K_total)}}}{{{fmt_number(K_planta)}}} = {fmt_number(Kc)}")

    # 5. Parâmetros Finais do Controlador (Kp, Kd, Ki)
    st.markdown("#### 5) Determinar os parâmetros finais do controlador ($K_p$, $K_d$, $K_i$)")
    st.markdown("Expandindo a função de transferência do controlador para a sua forma paralela canónica:")
    
    if ctrl_type == "PD":
        Kp = Kc * z_val
        Kd = Kc
        st.latex(rf"G_c(s) = K_c(s + z) = {fmt_number(Kc)}(s + {fmt_number(z_val)}) = {fmt_number(Kd)}s + {fmt_number(Kp)}")
        st.latex(rf"K_p = K_c \cdot z = {fmt_number(Kp)}")
        st.latex(rf"K_d = K_c = {fmt_number(Kd)}")
    
    elif ctrl_type == "PI":
        Kp = Kc
        Ki = Kc * z_val
        st.latex(rf"G_c(s) = K_c \frac{{(s + z)}}{{s}} = \frac{{{fmt_number(Kc)}(s + {fmt_number(z_val)})}}{{s}} = {fmt_number(Kp)} + \frac{{{fmt_number(Ki)}}}{{s}}")
        st.latex(rf"K_p = K_c = {fmt_number(Kp)}")
        st.latex(rf"K_i = K_c \cdot z = {fmt_number(Ki)}")
        
    else: # PID
        Kp = 2 * Kc * z_val
        Kd = Kc
        Ki = Kc * (z_val**2)
        st.latex(rf"G_c(s) = K_c \frac{{(s + z)^2}}{{s}} = \frac{{{fmt_number(Kc)}(s^2 + {fmt_number(2*z_val)}s + {fmt_number(z_val**2)})}}{{s}}")
        st.latex(rf"G_c(s) = {fmt_number(Kd)}s + {fmt_number(Kp)} + \frac{{{fmt_number(Ki)}}}{{s}}")
        st.latex(rf"K_p = 2 \cdot K_c \cdot z = {fmt_number(Kp)}")
        st.latex(rf"K_d = K_c = {fmt_number(Kd)}")
        st.latex(rf"K_i = K_c \cdot z^2 = {fmt_number(Ki)}")

    # 6. Simulação e Ajuste Fino
    st.markdown("#### 6) Simular o sistema e observar a resposta")
    
    adj_col1, adj_col2 = st.columns(2)
    with adj_col1:
        kc_fine = st.slider("Ganho ($K_c$)", min_value=0.0, max_value=float(Kc*4), value=float(Kc), step=0.05)
    with adj_col2:
        z_fine = st.slider("Zero ($z$)", min_value=0.0, max_value=float(z_val*4), value=float(z_val), step=0.05)

    if ctrl_type in ["PD", "PI"]:
        num_c_fine = [1, z_fine]
    else:
        num_c_fine = clean_coeffs(np.polymul([1, z_fine], [1, z_fine]))
        
    den_c_fine = [1, 0] if ctrl_type in ["PI", "PID"] else [1]
        
    G_planta = ct.TransferFunction(num_g, den_g)
    H_sensor = ct.TransferFunction(num_h, den_h)
    sys_cl_uncomp = ct.feedback(G_planta, H_sensor)
    
    G_controlador = ct.TransferFunction(num_c_fine, den_c_fine) * kc_fine
    sys_cl_comp = ct.feedback(G_controlador * G_planta, H_sensor)

    # Determinar janela temporal de simulação dinâmica em função da raiz
    t_sim_end = max(10, 4 / abs(sd.real) * 2) if abs(sd.real) > 1e-4 else 20
    t_eval = np.linspace(0, float(t_sim_end), 1000)
    
    t_uncomp, y_uncomp = ct.step_response(sys_cl_uncomp, T=t_eval)
    t_comp, y_comp = ct.step_response(sys_cl_comp, T=t_eval)

    mp_real = (max(y_comp) - 1.0) * 100 if max(y_comp) > 1 else 0
    tolerance = 0.05
    if input_method == "Métricas de Desempenho (Mp e ts)" and crit == "2%":
        tolerance = 0.02
        
    settled_indices = np.where(np.abs(y_comp - 1.0) > tolerance)[0]
    ts_real = t_comp[settled_indices[-1]] if len(settled_indices) > 0 else 0

    fig_time, ax_time = plt.subplots(figsize=(9, 5.5))
    setup_axis(ax_time)
    
    ax_time.plot(t_uncomp, y_uncomp, color='red', linewidth=2.5, label="Sem Controlador")
    ax_time.plot(t_comp, y_comp, color='blue', linewidth=2.5, label="Com Controlador")
    
    ax_time.scatter([t_comp[np.argmax(y_comp)]], [max(y_comp)], color="red", zorder=5)
    ax_time.annotate(f"$M_P$ = {mp_real:.2f}%", (t_comp[np.argmax(y_comp)], max(y_comp)), xytext=(0, 10), textcoords="offset points", ha='center')
    
    if ts_real > 0:
        ax_time.axvline(ts_real, color='red', linestyle='--', alpha=0.3)
        ax_time.annotate(f"$t_s$ = {ts_real:.2f}seg.", (ts_real, 0.8), xytext=(5, 0), textcoords="offset points")
        
    ax_time.axhline(1, color='gray', linestyle='-')
    ax_time.axhline(1 + tolerance, color='cyan', linestyle='--', alpha=0.5)
    ax_time.axhline(1 - tolerance, color='cyan', linestyle='--', alpha=0.5)
    
    ax_time.set_xlabel("Tempo")
    ax_time.set_ylabel("Resposta")
    ax_time.set_title("Resposta no Tempo")
    ax_time.legend()
    st.pyplot(fig_time)


def step_5():
    st.markdown("### 5. Aproximação Discretizada")
    st.markdown("A discretização é feita pela substituição algébrica do operador $s$ e posterior expansão dos polinómios.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Função Contínua $F(s)$ a discretizar:**")
        num_input = st.text_input("Numerador de F(s)", "7.58 30.32 22.74", key="num_disc")
        den_input = st.text_input("Denominador de F(s)", "1 9.81 31.67 56.12 28.1", key="den_disc")
    with col2:
        st.markdown("**Parâmetros de Amostragem:**")
        T = st.number_input("Período de amostragem T (s)", value=1.0, step=0.1)
        method = st.selectbox("Método de Discretização", [
            "Euler Direto / Avanço (Forward)",
            "Euler Indireto / Atraso (Backward)",
            "Tustin (Bilinear)"
        ])
        
    try:
        num_coeffs = parse_coeffs(num_input, "Numerador")
        den_coeffs = parse_coeffs(den_input, "Denominador")
    except Exception as e:
        status_line(str(e), "warning")
        return
        
    s, z = sp.symbols('s z')
    
    Num_s = sum(float(c) * s**(len(num_coeffs)-1-i) for i, c in enumerate(num_coeffs))
    Den_s = sum(float(c) * s**(len(den_coeffs)-1-i) for i, c in enumerate(den_coeffs))
    
    st.markdown("---")
    st.markdown("**I. Função de transferência no domínio contínuo:**")
    st.latex(rf"F(s) = \frac{{{fmt_poly(num_coeffs, digits=5)}}}{{{fmt_poly(den_coeffs, digits=5)}}}")
    
    if method == "Euler Direto / Avanço (Forward)":
        eq_s = (z - 1) / T
        st.markdown("**II. Aplicando a substituição pelo método de Euler Direto:**")
        if abs(T - 1.0) < TOL:
            eq_latex = r"(z - 1)"
        else:
            eq_latex = rf"\left(\frac{{z - 1}}{{{fmt_number(T)}}}\right)"
            
    elif method == "Euler Indireto / Atraso (Backward)":
        eq_s = (z - 1) / (T * z)
        st.markdown("**II. Aplicando a substituição pelo método de Euler Indireto:**")
        if abs(T - 1.0) < TOL:
            eq_latex = r"\left(\frac{{z - 1}}{{z}}\right)"
        else:
            eq_latex = rf"\left(\frac{{z - 1}}{{{fmt_number(T)}z}}\right)"
            
    else:
        eq_s = (2/T) * ((z - 1)/(z + 1))
        st.markdown("**II. Aplicando a substituição pelo método de Tustin (Bilinear):**")
        eq_latex = rf"\left({fmt_number(2/T)}\frac{{z - 1}}{{z + 1}}\right)"
        
    st.latex(rf"s = {sp.latex(sp.nsimplify(eq_s, tolerance=1e-4))}")
    
    def generate_sub_str(coeffs, eq_str):
        degree = len(coeffs) - 1
        terms = []
        for i, c in enumerate(coeffs):
            if abs(c) < TOL:
                continue
            power = degree - i
            c_str = fmt_number(c)
            if power == 0:
                terms.append(c_str)
            elif power == 1:
                terms.append(rf"{c_str}{eq_str}")
            else:
                terms.append(rf"{c_str}{eq_str}^{{{power}}}")
        if not terms: return "0"
        return " + ".join(terms).replace("+ -", "- ")

    N_z_raw_str = generate_sub_str(num_coeffs, eq_latex)
    D_z_raw_str = generate_sub_str(den_coeffs, eq_latex)
    
    st.markdown("**III. Substituir $s$ na equação e manter os termos agrupados:**")
    st.latex(rf"N(z) = {N_z_raw_str}")
    st.latex(rf"D(z) = {D_z_raw_str}")
    
    F_s = Num_s / Den_s
    F_z_raw = F_s.subs(s, eq_s)
    F_z_simp = sp.cancel(sp.simplify(F_z_raw))
    n_z, d_z = sp.fraction(F_z_simp)
    
    d_poly = sp.Poly(d_z, z)
    lead_coeff = d_poly.LC()
    
    if lead_coeff != 0:
        n_z = sp.expand(n_z / lead_coeff)
        d_z = sp.expand(d_z / lead_coeff)
    else:
        n_z = sp.expand(n_z)
        d_z = sp.expand(d_z)
        
    n_z_eval = n_z.evalf(5)
    d_z_eval = d_z.evalf(5)
    
    st.markdown("**IV. Expansão algébrica completa dos polinómios $N(z)$ e $D(z)$:**")
    st.latex(rf"N(z) = {sp.latex(n_z_eval)}")
    st.latex(rf"D(z) = {sp.latex(d_z_eval)}")
    
    st.markdown("**V. Função de transferência discretizada final:**")
    st.latex(rf"F(z) = \frac{{{sp.latex(n_z_eval)}}}{{{sp.latex(d_z_eval)}}}")


STEPS = {
    "1. Polinômio característico": ("Escrever o polinômio característico", step_1),
    "2. Fatoração de P(s)": ("Fatorar P(s) em polos e zeros", step_2),
    "3. Polos e zeros": ("Assinalar polos e zeros no plano s", step_3),
    "4. Projeto de Controlador": ("Projeto de Controlador (PD, PI, PID)", step_4),
    "5. Discretização": ("Aproximação Discretizada (Euler/Tustin)", step_5),
}


if view == "Todos os passos":
    for index, (step_key, (title, func)) in enumerate(STEPS.items(), start=1):
        number = step_key.split(".")[0]
        show_step(number, title, func)
else:
    title, func = STEPS[view]
    number = view.split(".")[0]
    show_step(number, title, func)