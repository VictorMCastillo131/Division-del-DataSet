import os
import uuid
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from django.conf import settings
from django.shortcuts import render
from sklearn.model_selection import train_test_split

sns.set(style="darkgrid")

def safe_read_dataset(uploaded_file):
    """Intenta leer el archivo como csv; si falla, lo lee como whitespace-delimited sin header."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        # para archivos .txt .data u otros intentamos delimitador por espacios o comas
        try:
            return pd.read_csv(uploaded_file, sep=',')
        except Exception:
            return pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
    except Exception as e:
        # si todo falla, subimos la excepción para que la vista la muestre
        raise e

def choose_protocol_column(df):
    """
    Busca 'protocol_type'. Si no existe, busca una columna categórica (dtype object)
    o la columna con pocos valores únicos (< 30). Devuelve el nombre de la columna encontrada.
    """
    if 'protocol_type' in df.columns:
        return 'protocol_type'
    # buscar columnas tipo object
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        return obj_cols[0]
    # buscar columna con pocos valores únicos
    for col in df.columns:
        if df[col].nunique() <= 30:
            return col
    # fallback: la primera columna
    return df.columns[0]

def save_plot(fig, filename):
    path = os.path.join(settings.MEDIA_ROOT, 'graphs')
    os.makedirs(path, exist_ok=True)
    fullpath = os.path.join(path, filename)
    fig.savefig(fullpath, bbox_inches='tight')
    plt.close(fig)
    return settings.MEDIA_URL + 'graphs/' + filename

def plot_hist_count(series, title):
    fig, ax = plt.subplots(figsize=(6,4))
    # convert to str for consistent binning/counting if needed
    sns.histplot(series.astype(str), discrete=True, shrink=0.8, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    return fig

def home(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        try:
            df = safe_read_dataset(uploaded_file)
            # si no tiene nombres de columnas y pandas lo puso header=None, le ponemos nombres simples
            if df.columns.dtype == 'int64' or any([str(c).startswith('Unnamed:') for c in df.columns]):
                # renombrar columnas con indices si no hay header
                df.columns = [f"col_{i}" for i in range(df.shape[1])]
            # División
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
            train_set, val_set = train_test_split(train_set, test_size=0.125, random_state=42, shuffle=True)  # 0.125*0.8 ~ 0.1

            # elegir columna a graficar (protocol_type o la más adecuada)
            col = choose_protocol_column(df)

            # carpeta y nombres únicos para evitar caché (uuid)
            uid = uuid.uuid4().hex[:8]

            # 4 gráficas: df, train, test, val
            fig1 = plot_hist_count(df[col].dropna(), f"Distribución en DF - Columna: {col}")
            fname1 = f"dist_df_{uid}.png"
            context['plot_df'] = save_plot(fig1, fname1)
            context['title_df'] = "Distribución: dataset completo"
            context['desc_df'] = f"Histograma de la columna **{col}** sobre el dataset original."

            fig2 = plot_hist_count(train_set[col].dropna(), f"Distribución en TRAIN - Columna: {col}")
            fname2 = f"dist_train_{uid}.png"
            context['plot_train'] = save_plot(fig2, fname2)
            context['title_train'] = "Distribución: train set"
            context['desc_train'] = "Histograma de la misma columna en el conjunto de entrenamiento."

            fig3 = plot_hist_count(test_set[col].dropna(), f"Distribución en TEST - Columna: {col}")
            fname3 = f"dist_test_{uid}.png"
            context['plot_test'] = save_plot(fig3, fname3)
            context['title_test'] = "Distribución: test set"
            context['desc_test'] = "Histograma de la misma columna en el conjunto de prueba."

            fig4 = plot_hist_count(val_set[col].dropna(), f"Distribución en VAL - Columna: {col}")
            fname4 = f"dist_val_{uid}.png"
            context['plot_val'] = save_plot(fig4, fname4)
            context['title_val'] = "Distribución: validation set"
            context['desc_val'] = "Histograma de la misma columna en el conjunto de validación."

            # información adicional para mostrar
            context.update({
                "columns": list(df.columns),
                "total_rows": len(df),
                "train_rows": len(train_set),
                "val_rows": len(val_set),
                "test_rows": len(test_set),
                "sample_data": df.head(8).to_html(classes='table table-striped', index=False),
                "protocol_column": col,
            })

        except Exception as e:
            context['error'] = f"Error al procesar el archivo: {e}"

    return render(request, 'upload.html', context)
