import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import association_rules, apriori
import plotly.express as px
import os
import seaborn as sns
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="Data Analysis of Minger", page_icon=":shopping_cart:")

st.title(" :bar_chart: Data Analysis of Minger")
st.write(
    "Come along to explore the Data Analysis Conducted for Minger! ")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
#load dataset
df = pd.read_csv('Global_superstore_data.csv')
path= pd.read_csv('association_rules.csv')


# Sidebar used in the streamlit app 
st.sidebar.title('Filters')
regions = df['Region'].unique()
selected_region = st.sidebar.selectbox('Select Region', regions)

sub_categories = df['Sub-Category'].unique()
selected_sub_category = st.sidebar.selectbox('Select Sub-Category', sub_categories)

# Filter data based on selected region and sub-category
filtered_df = df[(df['Region'] == selected_region) & (df['Sub-Category'] == selected_sub_category)]

# Display filtered data
st.subheader('Filtered Data')
st.write(filtered_df)



# insights used 
st.header("Product Insights")

# Create pivot table for heatmap
pivot_table = path.pivot(index='antecedents', columns='consequents', values='lift')


#Visualisation 01 
# Display the heatmap
st.subheader("Heatmap of Association Rules")
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5)
plt.xlabel("Consequents")
plt.ylabel("Antecedents")
plt.title("Heatmap of Association Rules (Lift)")
heatmap_fig = plt.gcf()  # Get the current figure
st.pyplot(heatmap_fig)  # Display the heatmap



#Visualisation 02
# Define best selling products
best_selling_products = ['Bookcases', 'Chairs', 'Supplies', 'Tables']

# Filter data to include only best selling products
best_selling_data = df[df['Sub-Category'].isin(best_selling_products)]
sub_category_sales = best_selling_data.groupby('Sub-Category')['Sales'].sum().reset_index()
sub_category_sales = sub_category_sales.sort_values(by='Sales', ascending=False)



nodes = list(set(path['antecedents']).union(set(path['consequents'])))
node_indices = {node: idx for idx, node in enumerate(nodes)}
node_names = {idx: node for idx, node in enumerate(nodes)}
sources = [node_indices[source] for source in path['antecedents']]
targets = [node_indices[target] for target in path['consequents']]
values = path['support']

# Create a Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        label=[node_names[idx] for idx in range(len(nodes))]
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values
    )
)])

# Update layout
fig.update_layout(title_text="Sankey Diagram of Association Rules")

# Show the Sankey diagram
st.plotly_chart(fig)


#Visualisation 03

st.subheader('Total Sales by Best Selling Products')
st.markdown('Below is the total sales for each of the best selling products:')
st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
selected_view = st.radio('Select View:', ('Plain Text', 'Table'))

if selected_view == 'Plain Text':
    for idx, row in sub_category_sales.iterrows():
        st.write(f"{row['Sub-Category']}: ${row['Sales']}")
else:
    st.table(sub_category_sales)
    
import plotly.express as px
fig = px.bar(sub_category_sales, x='Sub-Category', y='Sales', title="Total Sales by Best Selling Products",
             labels={'Sales': 'Total Sales', 'Sub-Category': 'Sub-Category'})
fig.update_layout(xaxis_title="Sub-Category", yaxis_title="Total Sales")
st.plotly_chart(fig)
segment_profit = df.groupby('Segment')['Profit'].sum().reset_index()
category_profit = df.groupby('Category')['Profit'].sum().reset_index()
fig_segment = px.pie(segment_profit, values='Profit', names='Segment', title='Profit by Segment')
fig_category = px.pie(category_profit, values='Profit', names='Category', title='Profit by Product Category')
chart_selection = st.selectbox('Select Chart:', ('Profit by Segment', 'Profit by Product Category'))
if chart_selection == 'Profit by Segment':
    st.plotly_chart(fig_segment)
else:
    st.plotly_chart(fig_category)
    
#Visualisation 04
# Filter data to include only specified sub-categories
sub_categories = ['Bookcases', 'Chairs', 'Tables', 'Supplies']
filtered_df = df[df['Sub-Category'].isin(sub_categories)]


region_subcategory_sales = filtered_df.groupby(['Region', 'Sub-Category'])['Sales'].sum().reset_index()

st.sidebar.title('Filters')
selected_region = st.sidebar.selectbox("Select Region", region_subcategory_sales['Region'].unique())

filtered_data = region_subcategory_sales[region_subcategory_sales['Region'] == selected_region]
fig = px.bar(filtered_data, x='Sub-Category', y='Sales', color='Sub-Category',
             title=f"Sales by Sub-Category in {selected_region}",
             labels={'Sales': 'Total Sales', 'Sub-Category': 'Sub-Category'})
fig.update_layout(xaxis_title="Sub-Category", yaxis_title="Total Sales")
st.plotly_chart(fig)


#Visualisation 05
import plotly.graph_objects as go

def generate_time_series(sub_category, color=None):
    filtered_df = df[df['Sub-Category'] == sub_category]
    filtered_df['Quarter'] = pd.to_datetime(filtered_df['Ship Date']).dt.to_period('Q').astype(str)
    quarterly_sales = filtered_df.groupby('Quarter')['Sales'].sum().reset_index()
    fig = px.line(quarterly_sales, x='Quarter', y='Sales', title=f"Quarterly Sales Trends for {sub_category}",
                  labels={'Sales': 'Total Sales', 'Quarter': 'Quarter'}, color_discrete_sequence=[color])
    return fig


selected_sub_category = st.sidebar.selectbox("Select Sub-Category", ['Bookcases', 'Chairs', 'Tables', 'Supplies'])
fig_individual = generate_time_series(selected_sub_category)
st.plotly_chart(fig_individual)


sub_categories = ['Bookcases', 'Chairs', 'Tables', 'Supplies']
filtered_df = df[df['Sub-Category'].isin(sub_categories)]
filtered_df['Quarter'] = pd.to_datetime(filtered_df['Ship Date']).dt.to_period('Q').astype(str)
quarterly_sales = filtered_df.groupby(['Quarter', 'Sub-Category'])['Sales'].sum().reset_index()
fig = px.line(quarterly_sales, x='Quarter', y='Sales', color='Sub-Category',
              title="Quarterly Sales Trends for Specific Sub-Categories",
              labels={'Sales': 'Total Sales', 'Quarter': 'Quarter'})
fig.update_layout(xaxis_title="Quarter", yaxis_title="Total Sales")
st.plotly_chart(fig)