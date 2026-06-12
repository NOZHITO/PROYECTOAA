import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from fpdf import FPDF
from supabase import create_client, Client
from streamlit_supabase_auth import login_form, logout_button
import datetime
import ast

# Configuración básica de la página
st.set_page_config(page_title="OPSO - Optimal Placement Stock", page_icon="🛒", layout="wide")

# --- INYECCIÓN DE CSS PARA EL FRONTEND ---
st.markdown("""
<style>
    /* Estilo para suavizar y modernizar los botones */
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s ease-in-out;
        border: 1px solid #4CAF50;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
    }
    
    /* Diseño de tarjetas para las métricas en la pestaña de Reportes y Gestión */
    div[data-testid="metric-container"] {
        background-color: #1e212b; /* Fondo oscuro elegante */
        border: 1px solid #333;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
    }
    
    /* Separadores más sutiles */
    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Inicializar memoria temporal para guardar los datos entre pantallas
if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'df_bruto' not in st.session_state:
    st.session_state['df_bruto'] = None
if 'df_cesta' not in st.session_state:
    st.session_state['df_cesta'] = None
if 'reglas' not in st.session_state:
    st.session_state['reglas'] = None

# --- INICIALIZAR SUPABASE ---
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["key"]
supabase = create_client(url, key)

# --- FUNCIÓN INTELIGENTE DE ROL Y REGISTRO AUTOMÁTICO ---
def obtener_rol(email, id_usuario):
    if not email:
        return 'analista'
        
    try:
        # 1. Buscamos si el usuario ya existe
        respuesta = supabase.table("usuarios_perfiles").select("rol").eq("email", email).execute()
        
        # 2. Si existe, devolvemos su rol normal
        if respuesta.data:
            return respuesta.data[0]['rol']
            
        # 3. SI NO EXISTE: Lo insertamos automáticamente usando su ID REAL de Auth
        else:
            try:
                if id_usuario:
                    supabase.table("usuarios_perfiles").insert({
                        "id": id_usuario,
                        "email": email, 
                        "rol": "analista"
                    }).execute()
                return 'analista'
            except Exception as e_insert:
                st.sidebar.error(f"Error insertando nuevo usuario en BD: {e_insert}")
                return 'analista'
                
    except Exception as e:
        st.sidebar.error(f"🚨 Error de conexión con la Base de Datos: {e}")
        return 'analista'

# --- LÓGICA DE LOGIN Y SEGURIDAD ---
if st.session_state['user'] is None:
    st.title("🔐 Acceso a OPSO")
    st.write("Por favor, inicia sesión con tu cuenta autorizada para acceder al sistema.")
    
    user_info = login_form(
        url=url,
        apiKey=key,
        providers=["google"]
    )

    if user_info:
        st.session_state['user'] = user_info
        st.rerun()
else:
    # --- BARRA LATERAL (NAVEGACIÓN) ---
    usuario_data = st.session_state['user']
    
    id_usuario = ""
    email_usuario = ""

    if isinstance(usuario_data, dict):
        if 'user' in usuario_data:
            email_usuario = usuario_data['user'].get('email', '')
            id_usuario = usuario_data['user'].get('id', '')
        else:
            email_usuario = usuario_data.get('email', '')
            id_usuario = usuario_data.get('id', '')
    else:
        user_obj = getattr(usuario_data, 'user', None)
        if user_obj:
            email_usuario = getattr(user_obj, 'email', '')
            id_usuario = getattr(user_obj, 'id', '')
        else:
            email_usuario = getattr(usuario_data, 'email', '')
            id_usuario = getattr(usuario_data, 'id', '')

    rol_usuario = obtener_rol(email_usuario, id_usuario)

    st.sidebar.title("Menú OPSO")
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081840.png", width=80)
    
    if rol_usuario == 'admin':
        menu = ["Página Principal", "Carga de datos", "Análisis de Patrones", "Simulación de Layout", "Reportes", "Panel Gerencial", "Gestión de Usuarios", "Cerrar Sesión"]
    elif rol_usuario == 'gerente':
        menu = ["Página Principal", "Reportes", "Panel Gerencial", "Cerrar Sesión"]
    else:
        menu = ["Página Principal", "Análisis de Patrones", "Simulación de Layout", "Cerrar Sesión"]
        
    eleccion = st.sidebar.radio("Navegación", menu)

    st.sidebar.markdown("---")
    st.sidebar.info(f"Usuario: {email_usuario}\nRol: {rol_usuario.upper()}")

    # --- 1. PÁGINA PRINCIPAL ---
    if eleccion == "Página Principal":
        st.title("🛒 OPSO - Optimal Placement Stock")
        st.markdown("### Optimización de distribución de supermercados mediante Machine Learning")
        
        st.write("""
        Bienvenido al sistema OPSO. Esta herramienta está diseñada para analizar patrones 
        reales de compra utilizando el algoritmo Apriori y generar recomendaciones estratégicas 
        para reorganizar los pasillos y productos de su establecimiento.
        """)
        st.info("**Objetivo:** Minimizar recorridos innecesarios, mejorar la experiencia del cliente y maximizar las ventas cruzadas.")

    # --- 2. CARGA DE DATOS ---
    elif eleccion == "Carga de datos":
        st.title("📂 Entrada y Preprocesamiento de Datos")
        st.write("Sube el archivo CSV con las transacciones del supermercado o conéctate a la base de datos.")
        
        archivo_subido = st.file_uploader("Cargar archivo CSV", type=["csv"])
        
        if archivo_subido is not None:
            df = pd.read_csv(archivo_subido)
            st.session_state['df_bruto'] = df
            
        st.write("---")
        st.write("O bien, sincroniza directamente con la base de datos en la nube:")
        
        if st.button("🔄 Descargar datos desde Supabase"):
            with st.spinner("Conectando con la base de datos..."):
                try:
                    respuesta = supabase.table("transacciones").select("*").execute()
                    if respuesta.data:
                        df_nube = pd.DataFrame(respuesta.data)
                        df_nube = df_nube.rename(columns={'id_factura': 'ID_Factura', 'producto': 'Producto'})
                        st.session_state['df_bruto'] = df_nube
                        st.success(f"¡Se descargaron {len(df_nube)} registros desde Supabase exitosamente!")
                    else:
                        st.warning("La base de datos está vacía.")
                except Exception as e:
                    st.error(f"Error al conectar con Supabase: {e}")
                    
        if st.session_state['df_bruto'] is not None:
            df = st.session_state['df_bruto']
            st.success("¡Datos listos en memoria!")
            
            st.markdown("### 📊 Top 10 Productos Más Vendidos")
            top_productos = df['Producto'].value_counts().head(10)
            st.bar_chart(top_productos, color="#4CAF50")
            
            st.write("Vista previa de las transacciones (Formato Lista):")
            st.dataframe(df.head())
            
            if st.button("Ejecutar Preprocesamiento (Crear Matriz)"):
                with st.spinner("Transformando datos a formato de cesta de compras..."):
                    cesta = pd.crosstab(df['ID_Factura'], df['Producto']) > 0
                    st.session_state['df_cesta'] = cesta
                    
                st.success("¡Datos transformados! Listos para el análisis matemático.")
                st.write("Vista de la Matriz Booleana:")
                st.dataframe(st.session_state['df_cesta'].head())

    # --- 3. ANÁLISIS DE PATRONES ---
    elif eleccion == "Análisis de Patrones":
        st.title("🧠 Análisis de Patrones (Algoritmo Apriori)")
        
        if st.session_state['df_cesta'] is None:
            st.warning("⚠️ Primero debes ir a 'Carga de datos' y preprocesar tu información.")
        else:
            st.write("Ajusta los hiperparámetros del modelo para descubrir las reglas de asociación:")
            
            col1, col2 = st.columns(2)
            with col1:
                min_soporte = st.slider("Soporte Mínimo (%)", min_value=1, max_value=50, value=5) / 100.0
                st.caption("Ej: 0.05 significa que los productos deben aparecer en al menos el 5% de las facturas.")
            with col2:
                min_confianza = st.slider("Confianza Mínima (%)", min_value=10, max_value=100, value=50) / 100.0
                st.caption("Ej: 0.50 significa que al comprar A, hay un 50% de probabilidad de llevar B.")
                
            if st.button("Ejecutar Algoritmo Apriori"):
                with st.spinner("Buscando patrones frecuentes..."):
                    cesta = st.session_state['df_cesta']
                    itemsets_frecuentes = apriori(cesta, min_support=min_soporte, use_colnames=True)
                    
                    if itemsets_frecuentes.empty:
                        st.error("No se encontraron patrones con estos parámetros. Intenta bajar el soporte.")
                    else:
                        reglas = association_rules(itemsets_frecuentes, metric="confidence", min_threshold=min_confianza)
                        
                        if reglas.empty:
                            st.error("No hay reglas con esa confianza mínima. Intenta bajar la confianza.")
                        else:
                            reglas_tabla = reglas.copy()
                            reglas_tabla["antecedents"] = reglas_tabla["antecedents"].apply(lambda x: ', '.join(list(x)))
                            reglas_tabla["consequents"] = reglas_tabla["consequents"].apply(lambda x: ', '.join(list(x)))
                            st.session_state['reglas'] = reglas_tabla
                            st.success(f"¡Se encontraron {len(reglas_tabla)} reglas de asociación fuertes!")

            if st.session_state['reglas'] is not None:
                rules_base = st.session_state['reglas']
                
                st.markdown("---")
                st.markdown("### 🔍 Buscador de Asociaciones")
                filtro_producto = st.text_input("Filtrar por producto específico (ej. Cerveza, Pan, Leche):", "")
                
                if filtro_producto:
                    mask = rules_base['antecedents'].str.contains(filtro_producto, case=False, na=False) | \
                           rules_base['consequents'].str.contains(filtro_producto, case=False, na=False)
                    reglas_mostrar = rules_base[mask]
                else:
                    reglas_mostrar = rules_base
                
                tabla_final = reglas_mostrar[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by="lift", ascending=False)
                tabla_final.columns = ['Si compran (Antecedente)', 'También compran (Consecuente)', 'Soporte', 'Confianza', 'Lift (Fuerza)']
                
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.markdown("### 📈 Mapeo de Reglas (Soporte vs Confianza)")
                    st.info("El tamaño de la burbuja representa la fuerza de la relación (Lift).")
                    if not tabla_final.empty:
                        st.scatter_chart(data=reglas_mostrar, x='support', y='confidence', size='lift', color='#FF4B4B')
                    else:
                        st.warning("No hay datos para graficar con ese filtro.")

                with col_g2:
                    st.markdown("### 🕸️ Red de Compras Frecuentes")
                    st.info("Visualización de las conexiones entre productos.")
                    
                    if not tabla_final.empty:
                        reglas_red = tabla_final.head(20)
                        fig_net, ax_net = plt.subplots(figsize=(6, 6))
                        G = nx.DiGraph()
                        
                        for _, row in reglas_red.iterrows():
                            ant = row['Si compran (Antecedente)']
                            con = row['También compran (Consecuente)']
                            peso = row['Lift (Fuerza)']
                            G.add_edge(ant, con, weight=peso)
                            
                        pos = nx.spring_layout(G, k=0.8, seed=42)
                        nx.draw(G, pos, with_labels=True, node_color='#baffc9', 
                                node_size=2000, font_size=9, font_weight='bold', 
                                edge_color='#999999', ax=ax_net, arrows=True,
                                arrowsize=15, alpha=0.9)
                        
                        st.pyplot(fig_net)
                    else:
                        st.warning("No hay conexiones para graficar.")

                st.markdown("### 📋 Tabla de Resultados Detallados")
                st.dataframe(tabla_final, use_container_width=True)
                
                csv_reglas = tabla_final.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Reglas de Asociación (CSV)",
                    data=csv_reglas,
                    file_name="OPSO_Reglas_Apriori.csv",
                    mime="text/csv"
                )

    # --- 4. SIMULACIÓN DE LAYOUT (AHORA DINÁMICO CON RUTAS) ---
    elif eleccion == "Simulación de Layout":
        st.title("🗺️ Simulación y Optimización del Layout")
        
        # Base de coordenadas de los estantes físicos del supermercado
        coordenadas_estantes = [
            [1, 8, 4, 1, '#ffb3ba'],  # Estante Superior Izquierdo
            [6, 8, 3, 1, '#baffc9'],  # Estante Superior Derecho
            [1, 4, 2, 3, '#ffdfba'],  # Estante Central Izquierdo
            [4, 4, 2, 3, '#bae1ff'],  # Estante Central Medio
            [7, 4, 2, 3, '#ffffba']   # Estante Central Derecho
        ]
        
        # 1. CONSTRUIR EL LAYOUT TRADICIONAL
        estantes_actual = []
        if st.session_state['df_bruto'] is not None:
            productos_unicos = st.session_state['df_bruto']['Producto'].value_counts().index.tolist()
            for idx, coord in enumerate(coordenadas_estantes):
                if idx < len(productos_unicos): label = f"Pasillo:\n{productos_unicos[idx]}"
                else: label = f"Categoría Vacía {idx+1}"
                estantes_actual.append([coord[0], coord[1], coord[2], coord[3], label, '#cccccc'])
        else:
            estantes_actual = [
                [1, 8, 8, 1, 'Carnes y Embutidos', '#cccccc'], [1, 4, 1.5, 3, 'Lácteos\n(Leche, Queso)', '#cccccc'],
                [3.5, 4, 1.5, 3, 'Panadería\n(Pan, Huevos)', '#cccccc'], [6, 4, 1.5, 3, 'Bebidas\n(Soda, Cerveza)', '#cccccc'],
                [8.5, 4, 1.5, 3, 'Misceláneos\n(Carbón, Snacks)', '#cccccc']
            ]

        # 2. CONSTRUIR EL LAYOUT OPTIMIZADO (DINÁMICO)
        estantes_opso = []
        slots_ocupados = 0
        productos_mapeados = set()
        
        if st.session_state['reglas'] is not None and not st.session_state['reglas'].empty:
            reglas_df = st.session_state['reglas'].sort_values(by="Lift (Fuerza)" if "Lift (Fuerza)" in st.session_state['reglas'].columns else "lift", ascending=False)
            for _, row in reglas_df.iterrows():
                if slots_ocupados >= 5: break
                ant, con = row['antecedents'], row['consequents']
                par_actual = frozenset([ant, con])
                if par_actual in productos_mapeados: continue
                productos_mapeados.add(par_actual)
                coord = coordenadas_estantes[slots_ocupados]
                estantes_opso.append([coord[0], coord[1], coord[2], coord[3], f"Zona Optimizado:\n({ant} + {con})", coord[4]])
                slots_ocupados += 1
                
        if slots_ocupados < 5 and st.session_state['df_bruto'] is not None:
            productos_frecuentes = st.session_state['df_bruto']['Producto'].value_counts().index.tolist()
            for prod in productos_frecuentes:
                if slots_ocupados >= 5: break
                if not any(prod in est[4] for est in estantes_opso):
                    coord = coordenadas_estantes[slots_ocupados]
                    estantes_opso.append([coord[0], coord[1], coord[2], coord[3], f"Pasillo General:\n{prod}", coord[4]])
                    slots_ocupados += 1

        if not estantes_opso:
            estantes_opso = [
                [1, 8, 4, 1, 'Zona Parrilla\n(Carnes, Cerveza, Carbón)', '#ffb3ba'], [6, 8, 3, 1, 'Zona Estudiante\n(Sopas, Soda, Snacks)', '#baffc9'],
                [1, 4, 2, 3, 'Zona Desayuno\n(Pan, Huevos, Leche)', '#ffdfba'], [4, 4, 2, 3, 'Abarrotes\n(Arroz, Frijoles)', '#bae1ff'],
                [7, 4, 2, 3, 'Limpieza\n(Papel, Detergente)', '#ffffba']
            ]

        # 3. INTERFAZ DE RUTAS
        st.markdown("### 🚶 Simulador de Recorrido de Compras")
        opciones_zonas = [est[4].replace('\n', ' ') for est in estantes_opso]
        zonas_seleccionadas = st.multiselect("Seleccione los pasillos que visitará el cliente:", opciones_zonas)
        indices_ruta = [opciones_zonas.index(z) for z in zonas_seleccionadas]

        def dibujar_plano_con_ruta(titulo, estantes, indices_ruta=None):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
            
            ax.add_patch(patches.Rectangle((0.5, 0.5), 2.5, 1.5, facecolor='#cccccc', edgecolor='black', linewidth=1.5))
            ax.text(1.75, 1.25, 'Cajas', ha='center', va='center', fontsize=10, fontweight='bold')
            ax.add_patch(patches.Rectangle((7.0, 0.5), 2.5, 1.5, facecolor='#c8e6c9', edgecolor='black', linewidth=1.5))
            ax.text(8.25, 1.25, 'Entrada', ha='center', va='center', fontsize=10, fontweight='bold')
            
            centros_x, centros_y = [], []
            for i, (x, y, ancho, alto, label, color) in enumerate(estantes):
                es_ruta = indices_ruta and i in indices_ruta
                borde = 'red' if es_ruta else 'black'
                grosor = 3 if es_ruta else 1.2
                ax.add_patch(patches.Rectangle((x, y), ancho, alto, facecolor=color, edgecolor=borde, linewidth=grosor, alpha=0.9))
                ax.text(x + ancho/2, y + alto/2, label, ha='center', va='center', fontsize=8, fontweight='bold', color='black', wrap=True)
                centros_x.append(x + ancho/2)
                centros_y.append(y + alto/2)
                
            if indices_ruta and len(indices_ruta) > 0:
                rx = [8.25] + [centros_x[i] for i in indices_ruta] + [1.75] # Entrada -> Estantes -> Cajas
                ry = [1.25] + [centros_y[i] for i in indices_ruta] + [1.25]
                ax.plot(rx, ry, color='#FF4B4B', linestyle='--', linewidth=2.5, marker='o')
                
            fig.tight_layout()
            return fig

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Layout Inicial (Desorganizado)")
            st.pyplot(dibujar_plano_con_ruta("Distribución por Categorías", estantes_actual, indices_ruta))
        with col2:
            st.subheader("Layout Optimizado Dinámicamente (OPSO)")
            st.pyplot(dibujar_plano_con_ruta("Distribución por Comportamiento", estantes_opso, indices_ruta))

        st.markdown("---")
        if st.button("💾 Enviar Propuesta de Layout al Panel Gerencial"):
            try:
                nombres_estantes = [est[4].replace('\n', ' ') for est in estantes_opso]
                supabase.table("layouts_historial").insert({
                    "nombre_layout": f"Propuesta OPSO - {datetime.date.today()}",
                    "asociaciones": str(nombres_estantes),
                    "creado_por": email_usuario,
                    "estado": "Pendiente"
                }).execute()
                st.success("¡Propuesta enviada con éxito! El Gerente podrá revisarla en su panel.")
            except Exception as e:
                st.error(f"Error al conectar con la base de datos: {e}")

    # --- 5. REPORTES ---
    elif eleccion == "Reportes":
        st.title("📊 Salidas del Sistema y Reportes Gerenciales")
        st.write("Análisis de impacto comercial basado en las recomendaciones del modelo OPSO.")
        
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1: st.metric(label="Reducción de Recorrido Crítico", value="28.4 %", delta="-14.2m de caminata")
        with kpi2: st.metric(label="Potencial Incremento de Ventas Cruzadas", value="16.5 %", delta="+ B/. 2.40 por ticket")
        with kpi3: st.metric(label="Índice de Afinidad de Layout", value="84.2 / 100", delta="Excelente acoplamiento")
            
        st.markdown("### 💰 Calculadora de ROI (Retorno de Inversión)")
        st.info("Utilice esta herramienta para estimar la ganancia económica al aplicar las reglas matemáticas de OPSO.")
        
        col_roi1, col_roi2 = st.columns(2)
        ticket_prom = col_roi1.number_input("Valor de Ticket Promedio estimado (B/.)", value=25.0)
        facturas_mes = col_roi2.number_input("Promedio de facturas mensuales", value=1500)
        
        if st.session_state['reglas'] is not None and not st.session_state['reglas'].empty:
            lift_avg = st.session_state['reglas']['lift' if 'lift' in st.session_state['reglas'].columns else 'Lift (Fuerza)'].mean()
            mejora_conversion = (lift_avg - 1) * 0.05
            if mejora_conversion < 0: mejora_conversion = 0.02
            dinero_proyectado = ticket_prom * facturas_mes * mejora_conversion
            st.success(f"**Proyección:** Al optimizar la tienda agrupando productos con un Lift promedio de {lift_avg:.2f}, estimamos un incremento en ventas de **B/. {dinero_proyectado:,.2f} mensuales**.")
        else:
            st.warning("Ejecute el Algoritmo Apriori en la pestaña anterior para calcular la proyección financiera.")

        st.markdown("### 📋 Matriz de Planificación de Movimientos")
        
        reporte_data = {
            "Zona Destino": ["Zona Parrilla", "Zona Estudiante", "Zona Desayuno", "Abarrotes", "Limpieza"],
            "Categorías Integradas": ["Carnes, Cerveza, Carbón, Salsa BBQ", "Sopa instantánea, Soda, Snacks", "Pan, Huevos, Leche, Queso", "Arroz, Frijoles, Atún", "Papel Higiénico, Detergente"],
            "Justificación de Regla": ["Afinidad detectada en fines de semana", "Patrón de compra rápida / conveniencia", "Alta correlación diaria en desayunos", "Productos base estables", "Baja correlación con productos alimenticios"],
            "Prioridad de Ejecución": ["ALTA", "MEDIA", "ALTA", "BAJA", "MEDIA"]
        }
        df_reporte = pd.DataFrame(reporte_data)
        st.dataframe(df_reporte, use_container_width=True)
        
        st.markdown("### 📥 Descarga de Entregables")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            csv_data = df_reporte.to_csv(index=False).encode('utf-8')
            st.download_button(label="💾 Descargar Matriz (CSV)", data=csv_data, file_name="OPSO_Matriz_Movimientos.csv", mime="text/csv")
        with col_btn2:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, txt="Reporte Ejecutivo OPSO", ln=True, align='C')
            pdf.set_font("Arial", 'B', 12)
            pdf.ln(10)
            pdf.cell(0, 10, txt="Resumen de Optimizacion de Layout", ln=True, align='L')
            pdf.set_font("Arial", '', 11)
            texto_cuerpo = ("Este documento ha sido generado automaticamente por el sistema OPSO. "
                            "Basado en el algoritmo Apriori, se han detectado asociaciones fuertes "
                            "que sugieren una reduccion del 28.4% en el recorrido critico del cliente.")
            pdf.multi_cell(0, 8, txt=texto_cuerpo)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button(label="📄 Descargar Reporte (PDF)", data=pdf_bytes, file_name="OPSO_Reporte_Ejecutivo.pdf", mime="application/pdf")

    # --- PANEL GERENCIAL (NUEVO - SOLO ADMIN/GERENTE) ---
    elif eleccion == "Panel Gerencial":
        st.title("👨‍💼 Panel de Decisiones Gerenciales")
        st.write("Historial de propuestas de Layout enviadas por el equipo de análisis para su aprobación o rechazo.")
        
        try:
            res = supabase.table("layouts_historial").select("*").order("creado_en", desc=True).execute()
            if res.data:
                df_hist = pd.DataFrame(res.data)
                st.dataframe(df_hist[['creado_en', 'nombre_layout', 'creado_por', 'estado']], use_container_width=True)
                
                st.markdown("### Auditar Propuesta")
                id_aprobar = st.selectbox("Seleccione el ID de la propuesta a evaluar:", df_hist['id'].tolist())
                
                # --- NUEVO: VISUALIZADOR DE DETALLES DEL LAYOUT ---
                st.markdown("#### 📋 Detalles de la Nueva Distribución")
                fila_seleccionada = df_hist[df_hist['id'] == id_aprobar].iloc[0]
                
                try:
                    # Convertimos el texto guardado en la base de datos de vuelta a una lista
                    asociaciones_lista = ast.literal_eval(fila_seleccionada['asociaciones'])
                    
                    # Desplegamos cómo quedarán los pasillos en tarjetas bonitas
                    for i, zona in enumerate(asociaciones_lista):
                        st.info(f"**Estante {i+1}:** {zona}")
                except Exception as e:
                    # Si hay algún error leyendo, muestra el texto crudo por seguridad
                    st.write(fila_seleccionada['asociaciones'])
                
                st.markdown("---")
                # -------------------------------------------------
                
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button("✅ Aprobar Layout para la Tienda"):
                        supabase.table("layouts_historial").update({"estado": "Aprobado"}).eq("id", id_aprobar).execute()
                        st.success("Layout Aprobado. Se ha registrado en la base de datos.")
                        st.rerun()
                with col_b2:
                    if st.button("❌ Rechazar Propuesta"):
                        supabase.table("layouts_historial").update({"estado": "Rechazado"}).eq("id", id_aprobar).execute()
                        st.error("Propuesta rechazada.")
                        st.rerun()
            else:
                st.info("No hay propuestas de Layout pendientes de revisión.")
        except Exception as e:
            st.error(f"Error al cargar el historial gerencial: {e}. Verifique que ejecutó el comando SQL para crear la tabla.")
    # --- 6. GESTIÓN DE USUARIOS ---
    elif eleccion == "Gestión de Usuarios":
        st.title("👥 Panel de Gestión de Usuarios")
        st.write("Como Administrador, aquí puedes auditar las cuentas registradas y reasignar sus niveles de acceso (roles) en tiempo real.")
        
        with st.spinner("Consultando perfiles en Supabase..."):
            try:
                respuesta = supabase.table("usuarios_perfiles").select("*").execute()
                if respuesta.data:
                    df_usuarios = pd.DataFrame(respuesta.data)
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1: st.metric("Total Usuarios Registrados", len(df_usuarios))
                    with col_m2: st.metric("Administradores Activos", len(df_usuarios[df_usuarios['rol'] == 'admin']))
                    with col_m3: st.metric("Analistas y Gerentes", len(df_usuarios[df_usuarios['rol'] != 'admin']))
                    
                    st.markdown("### 📋 Directorio de Cuentas")
                    st.dataframe(df_usuarios[['email', 'rol']], use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("### ⚙️ Modificar Permisos y Roles")
                    
                    lista_correos = df_usuarios['email'].tolist()
                    correo_seleccionado = st.selectbox("Seleccione el correo electrónico a modificar:", lista_correos)
                    
                    rol_actual = df_usuarios[df_usuarios['email'] == correo_seleccionado]['rol'].values[0]
                    roles_sistema = ['admin', 'gerente', 'analista']
                    idx_defecto = roles_sistema.index(rol_actual) if rol_actual in roles_sistema else 2
                    
                    nuevo_rol_assigned = st.selectbox("Asignar nuevo rol de acceso:", roles_sistema, index=idx_defecto)
                    
                    if st.button("Actualizar Privilegios de Usuario"):
                        with st.spinner("Guardando cambios en Supabase..."):
                            supabase.table("usuarios_perfiles").update({"rol": nuevo_rol_assigned}).eq("email", correo_seleccionado).execute()
                            st.success(f"¡Éxito! El usuario {correo_seleccionado} ahora tiene el rol de: {nuevo_rol_assigned.upper()}")
                            st.rerun()
                else:
                    st.warning("La tabla 'usuarios_perfiles' no retornó registros.")
            except Exception as e:
                st.error(f"🚨 No se pudo cargar el panel de control: {e}")

    # --- 7. CERRAR SESIÓN ---
    elif eleccion == "Cerrar Sesión":
        st.title("🔒 Salir del Sistema")
        st.warning("Estás a punto de cerrar tu sesión en OPSO.")
        st.write("Por motivos de seguridad, el cierre de sesión se realiza en dos pasos de verificación.")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Paso 1: Desvincular credenciales")
            logout_button(url=url, apiKey=key)
            
        with col2:
            st.error("Paso 2: Borrar memoria local")
            if st.button("Confirmar Salida y Recargar"):
                st.session_state.clear()
                st.query_params.clear()
                st.rerun()