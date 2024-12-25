import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas
import panel as pn

# Activer les extensions nécessaires
hvplot.extension('bokeh')
pn.extension()

# Charger les données
path = "C:\\ProjPy\\Unige_DSMLAI_2024_10\\Projet_Matteo_Musumeci\\predictive_maintenance.csv"
dataraw = pd.read_csv(path)

# Extraire uniquement les lettres M, L, H depuis Product ID
dataraw['Product_Category'] = dataraw['Product ID'].str.extract(r'([MLH])')

# Filtrer les lignes avec des catégories valides (M, L, H)
dataraw = dataraw[dataraw['Product_Category'].isin(['M', 'L', 'H'])]

# Retirer les colonnes inutiles
dataraw = dataraw.drop(columns=['UDI', 'Product ID', 'Type'])
dataraw['Temp_Diff'] = dataraw['Process temperature [K]'] - dataraw['Air temperature [K]']

# Vérifier les données après nettoyage
print(dataraw.head(15))
print(dataraw.shape)
print(dataraw.describe())
print(dataraw.columns)
print(pd.unique(dataraw["Failure Type"]))

# --- GRAPHIQUES INTERACTIFS ---

# 1. Histogramme interactif avec hvPlot
plot1 = dataraw.hvplot.hist(y="Torque [Nm]", bins=30, title="Distribution de Torque [Nm]")

# 2. Nuage de points interactif avec hvPlot
plot2 = dataraw.hvplot.scatter(x='Rotational speed [rpm]', y='Torque [Nm]', title="Vitesse vs Couple")

# 3. Histogramme interactif pour Temp_Diff
plot3 = dataraw.hvplot.hist(y="Temp_Diff", bins=30, title="Distribution de la Différence de Température")

# --- GRAPHIQUES STATIQUES SEABORN ---

# 4. Pairplot statique avec Seaborn
sns_plot = sns.pairplot(dataraw.select_dtypes(include='number'))
sns_plot.fig.savefig("pairplot.png")
pairplot_panel = pn.pane.Image("pairplot.png", width=600, height=600)

# 5. Heatmap avec Seaborn (colonnes numériques uniquement)
numeric_data = dataraw.select_dtypes(include='number')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
plt.title("Heatmap des Corrélations")
plt.savefig("heatmap.png")
heatmap_panel = pn.pane.Image("heatmap.png", width=600, height=600)

# 6. Boxplot avec les catégories M, L, H et légende améliorée
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=dataraw, x='Target', y='Tool wear [min]', hue='Product_Category', ax=ax)
plt.title('Tool Wear par Target et Catégories M, L, H', fontsize=14)
plt.xlabel('Échec de la Machine (Target)', fontsize=12)
plt.ylabel('Tool wear [min]', fontsize=12)
legend = ax.legend(title='Catégorie de Produit (M, L, H)', fontsize=10, title_fontsize=12, loc='upper right', bbox_to_anchor=(1.2, 1))
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')
ax.grid(True, linestyle='--', alpha=0.5)
plt.savefig("boxplot_tool_wear_MLH.png", bbox_inches='tight')
boxplot_product_MLH_panel = pn.pane.Image("boxplot_tool_wear_MLH.png", width=600, height=600)

# 7. Scatterplot avec Hue et Taille
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(data=dataraw, x='Rotational speed [rpm]', y='Torque [Nm]', hue='Target', size='Tool wear [min]', palette='coolwarm', ax=ax)
plt.title('Relation entre Rotational Speed, Torque et Tool Wear avec Target')
plt.savefig("scatterplot_rotational_torque_wear.png")
scatterplot_advanced_panel = pn.pane.Image("scatterplot_rotational_torque_wear.png", width=600, height=600)

# 8. Boxplot de Torque [Nm] par Target
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=dataraw, x='Target', y='Torque [Nm]', ax=ax)
plt.title('Distribution de Torque en Fonction de Target')
plt.savefig("boxplot_torque_target.png")
boxplot_torque_target_panel = pn.pane.Image("boxplot_torque_target.png", width=600, height=600)

# 9. Scatterplot simple entre Tool wear et Temp_Diff
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(data=dataraw, x='Temp_Diff', y='Tool wear [min]', hue='Target', palette='coolwarm', ax=ax)
plt.title('Relation entre Différence Température et Tool Wear par Target')
plt.savefig("scatterplot_tempdiff_toolwear.png")
scatterplot_tempdiff_toolwear_panel = pn.pane.Image("scatterplot_tempdiff_toolwear.png", width=600, height=600)

# --- GRAPHIQUES FAILURE TYPE ---

# 10. Boxplot Tool wear par Failure Type
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=dataraw, x='Failure Type', y='Tool wear [min]')
plt.title('Distribution de Tool Wear par Type de Panne')
plt.xticks(rotation=45)
plt.savefig("boxplot_toolwear_failure.png")
boxplot_toolwear_failure_panel = pn.pane.Image("boxplot_toolwear_failure.png", width=600, height=600)

# 11. Histogramme Torque par Failure Type
fig, ax = plt.subplots(figsize=(12, 8))
sns.histplot(data=dataraw, x='Torque [Nm]', hue='Failure Type', multiple='stack')
plt.title('Distribution de Torque par Type de Panne')
plt.savefig("histplot_torque_failure.png")
histplot_torque_failure_panel = pn.pane.Image("histplot_torque_failure.png", width=600, height=600)

# 12. Scatterplot Rotational speed vs Torque par Failure Type
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(data=dataraw, x='Rotational speed [rpm]', y='Torque [Nm]', hue='Failure Type', style='Failure Type')
plt.title('Relation entre Vitesse de Rotation et Torque par Type de Panne')
plt.savefig("scatterplot_failure_type.png")
scatterplot_failure_type_panel = pn.pane.Image("scatterplot_failure_type.png", width=600, height=600)

# --- TABLEAU DE BORD PANEL ---

dashboard = pn.Column(
    "# 🛠️ **Analyse du Dataset Predictive Maintenance**",
    pn.Row(plot1, plot2, plot3),
    pn.Row(pairplot_panel, heatmap_panel),
    pn.Row(boxplot_product_MLH_panel, scatterplot_advanced_panel),
    pn.Row(boxplot_toolwear_failure_panel, histplot_torque_failure_panel),
    pn.Row(scatterplot_failure_type_panel, scatterplot_tempdiff_toolwear_panel)
)

# Lancer le serveur Panel
if __name__ == '__main__':
    pn.serve(dashboard)
