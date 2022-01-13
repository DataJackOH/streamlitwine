import pandas as pd 
import streamlit as st 
import numpy as np
#example below is in the docs.
from collections import defaultdict
import requests
import json
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import plotly.express as px


##function to generate wordcloud from dataframe

url = 'https://raw.githubusercontent.com/DataJackOH/streamlitwine/f5a1045342c4a0305291c3056d39da9ea71b2b8d/red-wine-pouring-from-a-bottle-ralphbaleno-2.jpg'

redwineglass = requests.get(url)


def generatewc(df):
    mask = np.array(Image.open(BytesIO(redwineglass.content)))
    wc = WordCloud(mask=mask, background_color="rgba(255, 255, 255, 0)", mode="RGBA",
                max_words=150, max_font_size=256,
                random_state=42, width=mask.shape[1],
                height=mask.shape[0])

    # use ImageColorGenerator to generate the colors from the image
    image_colors = ImageColorGenerator(mask)


    # generate the word cloud
    wc.generate_from_frequencies(df)




    # use the new colors to color the wordcloud
    wc.recolor(color_func = image_colors)

    #plot
 
    ##build figure of all 
    fig = plt.figure(figsize=(8, 6), dpi=80)

    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.tight_layout(pad=0)
    fig.patch.set_alpha(0)
   
    return st.pyplot(fig)


st.title("Wine Review Analysis and NLP 🍷")


with st.expander("What is this tool?"):
    st.write("""
    This tool is a bit of fun, primarily to show off some nifty word-cloud functionality on a wine-review dataset.
    -   Custom colour and pattern word clouds
    -   More useful word clouds - beyond just a simple count of the most common words
    -   Interactive map visualising price/score data \n\n
    \n
    Building this involved some basic skills from the world of 'Natural Language Processing'; analysing  text data, which is a lot of fun!
     """)

    

with st.expander(label="Where's the data from?"):
    st.write("""The data was orginally scraped from the WineEnthusiast website, by a kaggle user and can be found [here](https://www.kaggle.com/zynicide/wine-reviews).

In total there are 150k reviews, and **all** of them are 'good' wines i.e. scored 80 or above. \n

However, they cover a range of countries and most interestingly, a wide range of price points - from $4 - $2000+ USD.""")




st.header('A 🍷 shaped word cloud')


with st.expander(label="What are the two word clouds?"):
    st.write("""We have grouped each wine into price categories from value ($4-$7 USD) to ridiculous ($200+ usd)

The left wordcloud (the bottle):
-   simple count of word frequency 
-   size of word = a count of times it word appears
-   common 'stop' words (the, and, it etc.. ) removed
***
The right wordcloud (the pouring bottle + glass):
-   much more interesting! 
-   a technique called 'log likelihood'
-   measuring words AND short phrases
-   size of word = how much more likely it is to appear for this price, relative to all other prices
-   limited to at least 50 occurences, so it's not ***too*** out there 
    """)


##allow interactive selection for word cloud

wordcloudchoice = st.selectbox("Pick the wine price range ", ["Extreme Value - $3-7", 'Value - $7-10','Midrange - $10-20','Premium - $20-50','Luxury - $50-100','Super Luxury - $100-200', 'Ridiculous - $200+'])


wordcloudsplit = wordcloudchoice.split('-')[0]

wordcloudselect = wordcloudsplit.lower().strip()

col1, col2 = st.columns(2)

##group data frame by price

url = 'https://raw.githubusercontent.com/DataJackOH/streamlitwine/main/wcdict.json'

resp = requests.get(url)
groupeddfprice = json.loads(resp.text)


url = 'https://raw.githubusercontent.com/DataJackOH/streamlitwine/main/33947791556638.jpg'

glassbottle = requests.get(url)


mask = np.array(Image.open(BytesIO(glassbottle.content)))
wc = WordCloud(mask=mask, background_color="rgba(255, 255, 255, 0)", mode="RGBA",
               max_words=150, max_font_size=256,
               random_state=42, width=mask.shape[1],
               height=mask.shape[0])

# use ImageColorGenerator to generate the colors from the image
image_colors = ImageColorGenerator(mask)

dictget = groupeddfprice.get(wordcloudselect)
wc.generate(' '.join(dictget))
#wc.generate(' '.join(groupeddfprice.get(wordcloudselect)))


# use the new colors to color the wordcloud
wc.recolor(color_func = image_colors)

##build figure of all 
fig1 = plt.figure(figsize=(50, 16), dpi=120)
plt.imshow(wc, interpolation="bilinear")
plt.axis('off')
plt.tight_layout(pad=0)
plt.rcParams["figure.figsize"]=(20,16)
fig1.patch.set_alpha(0)


with col1:
    st.pyplot(fig1)



##read in pre-prepared log wordcloud
t_bow_df = pd.read_csv('https://raw.githubusercontent.com/DataJackOH/streamlitwine/main/app_value_wordranking.csv')
t_bow_df = t_bow_df.set_index('Unnamed: 0')


# create a pandas Series of the top 4000 most frequent words
text=t_bow_df.loc[wordcloudselect].sort_values(ascending=False)


# create a dictionary Note: you could pass the pandas Series directoy into the wordcloud object
text2_dict=t_bow_df.loc[wordcloudselect].sort_values(ascending=False).to_dict()

with col2: 
    logwc = generatewc(text2_dict)

with col1:
    st.caption('A simple frequency word cloud')
    
with col2: 
    st.write("""
    
    
    
    
    """)
    
with col2:
    st.caption('Frequency, relative to all other prices')
    
   

st.caption("""Notice how the simpler word cloud contains a lot of similar words between prices - it looks like reviewers like to lean on the same set of words to describe all prices of wine.
-   Finish, Flavor, and Fruit tend to occur very frequently in most reviews , irrespective off price
    """)
st.caption("""With the right word cloud, we can immediately see how much more interesting a relative frequency is than a simple count:
-   Party, Buy, Basic and Bland are used relatively more often to describe lower priced wines
-   Unwind, Pipe, Tobacco, Vineyard, Opulence and Stunning used relatively more often to describe premium wines
    """)
st.write("***")

meanprice = pd.read_csv('https://raw.githubusercontent.com/DataJackOH/streamlitwine/main/meanprice.csv')
meanpoints = pd.read_csv('https://raw.githubusercontent.com/DataJackOH/streamlitwine/main/meanpoints.csv')
totalreviews = pd.read_csv('https://raw.githubusercontent.com/DataJackOH/streamlitwine/main/totalreviews.csv')

@st.cache
def group_map(y='points'):
    
    if y == 'Average Score':
       df_mean = meanpoints.copy()
    elif y == 'Average Price (USD)':
        df_mean = meanprice.copy()
    elif y == 'Total Reviews':
        df_mean = totalreviews.copy()

    
    if y == 'Average Score':
       data = 'points'
    elif y == 'Average Price (USD)':
        data = 'price'
    elif y == 'Total Reviews':
        data = 'totalreviews'


    fig = px.choropleth(df_mean, locationmode='country names',locations='country', color=data,
    color_continuous_scale=px.colors.sequential.Agsunset
    ,projection='mollweide',
    )

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

#map

st.subheader('Visualising 🍷 Geography')



with st.expander(label="What is this map?"):
    st.write("""The map is built from the same dataset as the reviews. 

We are using the wine production country to visualise three metrics:  
-   Average price in USD
-   The average review score of the wine
-   The total number of reviews
    """)

selectedmap = st.selectbox("Pick the metric you'd like to see on the map", ["Average Price (USD)", 'Average Score', 'Total Reviews'])



##create figure
fig = group_map(y=selectedmap)   

st.plotly_chart(fig)

 



st.caption("Please send me a message on [LinkedIn](https://www.linkedin.com/in/jack-o-hagan-349561a8/) if you have any questions, or want to learn more.")



st.caption("For those that want to see the code, check out the [GitHub repo](https://github.com/DataJackOH/streamlitwine).")

