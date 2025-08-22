import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import plotly.express as px
from datetime import datetime
from pandas.tseries.offsets import DateOffset

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title = 'Predicción de Renta por m2', layout = 'wide')

# --- AUTENTICACIÓN SIMPLE ---
PASSWORD = 'ModeloConquer!'
password_input = st.sidebar.text_input('Ingresa la clave', type = 'password')
if password_input != PASSWORD:
    st.warning('🔒 Ingresa la clave correcta para acceder a la aplicación.')
    st.stop()

# --- CARGA DE MODELO Y PIPELINE ---
@st.cache_resource
def load_models():
    pipeline = joblib.load('pipeline_modelo.pkl')
    modelo = joblib.load('modelo_final_xgboost.pkl')
    return pipeline, modelo

pipeline, modelo = load_models()

# --- CARGA DE DATASET MAESTRO ---
@st.cache_data
def load_dataset_maestro():
    fname_candidates = ['dataset_maestro_app.xlsx', 'dataset_maestro.xlsx']
    fname = next((f for f in fname_candidates if os.path.exists(f)), fname_candidates[-1])
    df = pd.read_excel(fname, sheet_name = 'dataset_maestro')
    df.drop(columns = ['mapeo.CONSTRUCCION'], errors = 'ignore', inplace = True)
    return df

df_maestro = load_dataset_maestro()

# --- CARGA DE CATALOGO ---
@st.cache_data
def load_cat_giro():
    return pd.read_csv('cat_giro.csv')

cat_giro = load_cat_giro()

# --- CONSTANTES ---
MAPEO_COLS = [
    'mapeo.UBICACION','mapeo.ANO_DE_APERTURA','mapeo.GLA','mapeo.OCUPACION','mapeo.AREA_RENTADA',
    'mapeo.AFLUENCIA','mapeo.NOI','mapeo.CONSTRUCCION','mapeo.SUPERFICIE_TOTAL','mapeo.CAJONES',
    'mapeo.NUMERO_INQUILINOS','mapeo.PLANTAS','mapeo.TIPO_DE_PLAZA','mapeo.REDES_SOCIALES_FOLLOWERS',
    'mapeo.SUPERMERCADO','mapeo.CINE','mapeo.GIMNASIO','mapeo.DEPARTAMENTAL','mapeo.COBRA_ESTACIONAMIENTO',
    'mapeo.AVENIDA_PRINCIPAL'
]

# Helper para labels visibles (quita el prefijo mapeo.)
def label_visible(col):
    return col.replace('mapeo.', '').replace('_', ' ')
    
# --- FORMULARIO / ARCHIVO DE ENTRADA ---
st.title('📈 Predicción de Renta por Metro Cuadrado')

st.markdown('Carga un archivo con los siguientes campos, o ingrésalos manualmente:')
st.code('PLAZA, LOCAL, NOMBRE, GIRO, SUPERFICIE, MXN_POR_M2, FECHA_INICIO, FECHA_FIN, UBICACION')

uploaded_file = st.file_uploader('Sube un archivo CSV o Excel', type = ['csv', 'xlsx'])

# -----------------------------------------------------------
# 1) MODO ARCHIVO
# -----------------------------------------------------------
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file)

# -----------------------------------------------------------
# 2) MODO MANUAL
# -----------------------------------------------------------
else:
    st.subheader('Captura manual')

    if 'plaza_confirmed' not in st.session_state:
        st.session_state.plaza_confirmed = False
        st.session_state.plaza_selected = None

    plaza_sel = st.selectbox('PLAZA', sorted(df_maestro['PLAZA'].dropna().unique()), index = 0)
    if st.button('OK'):
        st.session_state.plaza_confirmed = True
        st.session_state.plaza_selected = plaza_sel

    if not st.session_state.plaza_confirmed:
        st.stop()

    plaza = st.session_state.plaza_selected
    row = df_maestro[df_maestro['PLAZA'] == plaza].iloc[:1]
    defaults_mapeo = {c: None for c in MAPEO_COLS}
    if not row.empty:
        for c in MAPEO_COLS:
            if c in row.columns:
                defaults_mapeo[c] = row[c].iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        local = st.text_input('LOCAL', value = 'A101').upper()
        giro = st.selectbox('GIRO', sorted(cat_giro['GIRO'].dropna().unique()))
    with col2:
        nombre = st.text_input('NOMBRE', value = 'NUEVO LOCAL').upper()
        superficie = st.number_input('SUPERFICIE', min_value = 1.0, step = 1.0)
    with col3:
        mxn_m2 = st.number_input('MXN_POR_M2', min_value = 0.0, step = 10.0)

    col4, col5 = st.columns(2)
    with col4:
        fecha_inicio = st.date_input('FECHA_INICIO', value = datetime.today())
    with col5:
        fecha_fin = st.date_input('FECHA_FIN', value = datetime.today())

    st.markdown('Atributos mapeo (editables)')
    mapeo_values = {}
    # Primera columna debe iniciar con UBICACION
    for chunk_start in range(0, len(MAPEO_COLS), 4):
        cols = st.columns(4)
        for i, c in enumerate(MAPEO_COLS[chunk_start:chunk_start+4]):
            default_val = defaults_mapeo.get(c)
            if c in df_maestro.columns and pd.api.types.is_numeric_dtype(df_maestro[c]):
                val = cols[i].number_input(label_visible(c), value = float(default_val) if pd.notna(default_val) else 0.0)
            else:
                if c == 'mapeo.UBICACION':
                    ubicaciones = sorted(df_maestro['mapeo.UBICACION'].dropna().unique())
                    default_ubi = (row['mapeo.UBICACION'].iloc[0] if 'mapeo.UBICACION' in row.columns else (ubicaciones[0] if ubicaciones else ''))
                    val = cols[i].selectbox(label_visible(c), ubicaciones, index = (ubicaciones.index(default_ubi) if default_ubi in ubicaciones else 0))
                elif c == 'mapeo.TIPO_DE_PLAZA':
                    ubicaciones = sorted(df_maestro['mapeo.TIPO_DE_PLAZA'].dropna().unique())
                    default_ubi = (row['mapeo.TIPO_DE_PLAZA'].iloc[0] if 'mapeo.TIPO_DE_PLAZA' in row.columns else (ubicaciones[0] if ubicaciones else ''))
                    val = cols[i].selectbox(label_visible(c), ubicaciones, index = (ubicaciones.index(default_ubi) if default_ubi in ubicaciones else 0))
                else:
                    val = cols[i].text_input(label_visible(c), value = '' if default_val is None else str(default_val))
            mapeo_values[c] = val

    if st.button('Confirmar datos'):
        base_dict = {
            'PLAZA': plaza,
            'LOCAL': local,
            'NOMBRE': nombre,
            'GIRO': giro,
            'SUPERFICIE': superficie,
            'MXN_POR_M2': mxn_m2 if mxn_m2 > 0 else np.nan,
            'FECHA_INICIO': pd.to_datetime(fecha_inicio),
            'FECHA_FIN': pd.to_datetime(fecha_fin)
        }
        base_dict.update(mapeo_values)
        df_input = pd.DataFrame([base_dict])

    if 'df_input' in locals():
        if df_input.isnull().all(axis = 1).any():
            st.stop()
            
# -----------------------------------------------------------
# 3) PROCESAMIENTO SI HAY DATOS DE ENTRADA
# -----------------------------------------------------------
if 'df_input' in locals():

    # --- CRUCE INTELIGENTE CON DATASET MAESTRO ---
    # 3.1 Quitar del maestro todas las columnas que el usuario ya trae para evitar duplicados en el merge
    base_cols_a_ignorar = ['MXN_POR_M2', 'FECHA_INICIO', 'FECHA_FIN', 'LOCAL', 'NOMBRE', 'GIRO', 'SUPERFICIE']
    mapeo_en_entrada = [c for c in MAPEO_COLS if c in df_input.columns]
    cols_a_drop = [c for c in base_cols_a_ignorar + mapeo_en_entrada if c in df_maestro.columns]
    df_maestro_base = df_maestro.drop(columns = cols_a_drop)

    # 3.2 Merge con prioridad: (PLAZA + mapeo.UBICACION) → PLAZA → mapeo.UBICACION
    df_joined = None

    if all(col in df_input.columns for col in ['PLAZA', 'mapeo.UBICACION']) and \
       all(col in df_maestro_base.columns for col in ['PLAZA', 'mapeo.UBICACION']):
        df_joined = df_input.merge(df_maestro_base, on = ['PLAZA', 'mapeo.UBICACION'], how = 'left')

    if df_joined is None or df_joined.isnull().all(axis = 1).any():
        if 'PLAZA' in df_input.columns and 'PLAZA' in df_maestro_base.columns:
            df_joined = df_input.merge(df_maestro_base.drop_duplicates(subset = ['PLAZA']), on = ['PLAZA'], how = 'left')

    if df_joined is None or df_joined.isnull().all(axis = 1).any():
        if 'mapeo.UBICACION' in df_input.columns and 'mapeo.UBICACION' in df_maestro_base.columns:
            df_joined = df_input.merge(
                df_maestro_base.drop_duplicates(subset = ['mapeo.UBICACION']),
                on = ['mapeo.UBICACION'], how = 'left'
            )

    if df_joined is None:
        st.error('❌ No se puede hacer el cruce: faltan columnas clave en los datos.')
        st.stop()

    # 3.3 Limpieza y derivadas
    if not {'PLAZA','LOCAL','mapeo.UBICACION'}.issubset(df_joined.columns):
        st.error('❌ El cruce de datos no generó columnas clave. Verifica los campos ingresados.')
        st.stop()

    df = df_joined.drop_duplicates(subset = ['PLAZA', 'LOCAL', 'mapeo.UBICACION'], keep = 'first').copy()
    df['FECHA_INICIO'] = pd.to_datetime(df['FECHA_INICIO'], errors = 'coerce')
    df['FECHA_FIN'] = pd.to_datetime(df['FECHA_FIN'], errors = 'coerce')
    df['GIRO'] = df['GIRO'].fillna('SIN CLASIFICAR')

    # Variables derivadas
    df['DURACION_MESES'] = ((df['FECHA_FIN'] - df['FECHA_INICIO']).dt.days / 30.44).round(1)
    df['ESTA_VIGENTE'] = (df['FECHA_FIN'] >= pd.Timestamp.today()).astype(int)
    df['ANTIGÜEDAD_MESES'] = ((pd.Timestamp.today() - df['FECHA_INICIO']).dt.days / 30.44).round(1)
    df['ID_LOCAL'] = df['PLAZA'].astype(str) + ' - ' + df['LOCAL'].astype(str) + ' - ' + df['mapeo.UBICACION'].astype(str)
    df['ES_KIOSKO'] = (df['GIRO'].str.upper().str.strip() == 'KIOSKOS').astype(int)
    df['VINTAGE_INICIO'] = df['FECHA_INICIO'].dt.strftime('%Y%m').astype(int)
    df['VINTAGE_FIN'] = df['FECHA_FIN'].dt.strftime('%Y%m').astype(int)
    df['EXPIRA_24M'] = (df['FECHA_FIN'] <= (pd.Timestamp.today() + DateOffset(months = 24))).astype(int)

    def clasificar_tamano(m2):
        if m2 < 20:
            return 'KIOSKO'
        elif m2 <= 150:
            return 'SMALL'
        elif m2 <= 500:
            return 'MEDIUM'
        else:
            return 'LARGE'

    df['TAMANO_LOCAL'] = df['SUPERFICIE'].apply(clasificar_tamano)

    # Validación silenciosa de columnas esperadas por el pipeline
    if hasattr(pipeline, 'feature_names_in_'):
        expected_cols = set(pipeline.feature_names_in_)
        actual_cols = set(df.columns)
        missing_cols = expected_cols - actual_cols
        if missing_cols:
            st.stop()

    # --- PREDICCIÓN ---
    X_proc = pipeline.transform(df)
    y_pred_log = modelo.predict(X_proc)
    df['PREDICCIÓN_LOG'] = y_pred_log
    df['PREDICCIÓN_MXN_POR_M2'] = np.expm1(y_pred_log)

    # --- VISUALIZACIÓN DE RESULTADOS ---
    st.subheader('📋 Resultados de la Predicción')

    if len(df) == 1:
        renta_predicha = df['PREDICCIÓN_MXN_POR_M2'].iloc[0]
        st.success(f'💡 Predicción estimada: ${renta_predicha:,.2f} MXN por m²')

    # columnas_mostrar = ['PLAZA', 'PORTFOLIO', 'LOCAL', 'NOMBRE', 'GIRO', 'SUPERFICIE', 'RENTA', 'MXN_POR_M2', 'FECHA_INICIO', 'FECHA_FIN', 'mapeo.UBICACION', 'PREDICCIÓN_MXN_POR_M2']
    columnas_mostrar = list(df_input.columns) + ['PREDICCIÓN_MXN_POR_M2']
    columnas_presentes = [col for col in columnas_mostrar if col in df.columns]
    df_vista = df[columnas_presentes].copy()

    with st.expander('🔍 Ver tabla completa'):
        st.dataframe(df_vista.style.format({'PREDICCIÓN_MXN_POR_M2': '{:.2f}'}), height = 400)

    # --- DESCARGA ---
    def convertir_csv(df_):
        return df_.to_csv(index = False).encode('utf-8')

    st.download_button(
        label = '⬇️ Descargar resultados como CSV',
        data = convertir_csv(df),
        file_name = 'predicciones_renta.csv',
        mime = 'text/csv'
    )

    # --- GRÁFICOS ---
    fecha_actual = pd.Timestamp.today()
    df['MESES_RESTANTES'] = ((df['FECHA_FIN'] - fecha_actual).dt.days / 30.44).clip(lower = 0)

    # Renta de mercado estimada por GIRO (si no existe)
    if 'RENTA_MERCADO' not in df.columns:
        mediana_por_giro = df.groupby('GIRO')['MXN_POR_M2'].median()
        df['RENTA_MERCADO'] = df['GIRO'].map(mediana_por_giro)

    df['delta_rent_pct'] = (df['MXN_POR_M2'] / df['RENTA_MERCADO'] - 1) * 100
    df['delta_pred_pct'] = (df['PREDICCIÓN_MXN_POR_M2'] / df['RENTA_MERCADO'] - 1) * 100
    df['delta_gap'] = abs(df['delta_rent_pct'] - df['delta_pred_pct'])

    df_vigentes = df.copy()
    df_vigentes['RENTA_MERCADO'] = df_vigentes['RENTA_MERCADO'].fillna(df_vigentes['MXN_POR_M2'].median())

    # --- 1. GRÁFICO POR PLAZA: leyenda por PLAZA (no por PORTAFOLIO) ---
    if len(df_vigentes) > 1:
        g = df_vigentes.groupby('PLAZA')
        df_plaza = pd.DataFrame({
            'PLAZA': g.size().index,
            'SUPERFICIE': g['SUPERFICIE'].sum().values,
            'MESES_RESTANTES': g.apply(lambda s: np.average(s['MESES_RESTANTES'], weights = s['SUPERFICIE'])).values,
            'MXN_POR_M2': g.apply(lambda s: np.average(s['MXN_POR_M2'], weights = s['SUPERFICIE'])).values,
            'PREDICCIÓN_MXN_POR_M2': g.apply(lambda s: np.average(s['PREDICCIÓN_MXN_POR_M2'], weights = s['SUPERFICIE'])).values,
        })
        df_plaza['Delta PRX ponderado (1 - modelo / real)'] = 1 - (df_plaza['PREDICCIÓN_MXN_POR_M2'] / df_plaza['MXN_POR_M2'])

        fig_plaza = px.scatter(
            df_plaza,
            x = 'MESES_RESTANTES',
            y = 'Delta PRX ponderado (1 - modelo / real)',
            color = 'PLAZA',  # leyenda por PLAZA
            hover_name = 'PLAZA',
            hover_data = {
                'MXN_POR_M2': ':.2f',
                'PREDICCIÓN_MXN_POR_M2': ':.2f'
            },
            title = '📊 Rendimiento por Plaza - Comparación PRX Estimado vs Real (Ponderado)'
        )
        fig_plaza.add_hline(y = 0, line_dash = 'dash', line_color = 'gray')
        fig_plaza.add_vline(x = 24, line_dash = 'dash', line_color = 'gray')
        fig_plaza.update_traces(marker = dict(line = dict(width = 1, color = 'DarkSlateGrey')))
        fig_plaza.update_layout(showlegend = True)
        st.plotly_chart(fig_plaza, use_container_width = True)

    # --- 2. GRÁFICO POR LOCAL (sin agrupar) ---
    if len(df_vigentes) >= 1:
        fig_local = px.scatter(
            df_vigentes,
            x = 'MESES_RESTANTES',
            y = 'delta_gap',
            hover_name = 'NOMBRE',
            hover_data = {
                'PLAZA': True,
                'LOCAL': True,
                'GIRO': True,
                'delta_rent_pct': ':.2f',
                'delta_pred_pct': ':.2f'
            },
            color = 'GIRO',
            title = '🔍 Brecha por Local – Real vs Predicho vs Mercado',
            labels = {
                'MESES_RESTANTES': 'Meses hasta vencimiento',
                'delta_gap': 'Brecha real – predicha (%)'
            }
        )
        fig_local.add_hline(y = 0, line_dash = 'dash', line_color = 'gray')
        fig_local.add_vline(x = 24, line_dash = 'dash', line_color = 'gray')
        fig_local.update_traces(marker = dict(size = 10))
        fig_local.update_layout(showlegend = True)
        st.plotly_chart(fig_local, use_container_width = True)









