#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[10]:


columns_name=['user_id','item_id','rating','timestamp']
df=pd.read_csv('u.data',sep="\t",names=columns_name)


# In[11]:


df.head()


# In[9]:


df.shape


# In[12]:


df['user_id']


# In[13]:


df['user_id'].nunique()


# In[14]:


df['item_id'].nunique()


# In[15]:


movies_title=pd.read_csv('u.item',sep="\|",header=None)


# In[16]:


movies_title.shape


# In[19]:


movies_titles=movies_title[[0,1]]
movies_titles.columns=["item_id","title"]
movies_titles.head()


# In[20]:


df=pd.merge(df,movies_titles,on="item_id")


# In[21]:


df


# In[22]:


df.tail()


# In[26]:


ratings=pd.DataFrame(df.groupby('title').mean()['rating'])


# In[27]:


ratings.head()


# In[28]:


ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])


# #Create the recommander System

# In[29]:


df.head()


# In[42]:


moviemat=df.pivot_table(index="user_id",columns="title",values="rating")


# In[43]:


moviemat.head()


# In[44]:


starwars_user_ratings=moviemat['Star Wars (1977)']


# In[47]:


starwars_user_ratings.head(20)


# In[48]:


similar_to_starwars=moviemat.corrwith(starwars_user_ratings)


# In[49]:


similar_to_starwars


# In[50]:


corr_starwars=pd.DataFrame(similar_to_starwars,columns=['correlation'])


# In[51]:


corr_starwars.dropna(inplace=True)


# In[52]:


corr_starwars


# In[53]:


corr_starwars.head()


# In[54]:


corr_starwars.sort_values('correlation',ascending=False).head(10)


# In[55]:


ratings


# In[56]:


corr_starwars=corr_starwars.join(ratings['num of ratings'])


# In[57]:


corr_starwars


# In[58]:


corr_starwars.head()


# In[61]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('correlation',ascending=False)


# In[62]:


def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar_to_movie,columns=['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['num of ratings'])
    
    predictions=corr_movie[corr_movie['num of ratings']>100].sort_values('correlation',ascending=False)
    
    return predictions


# In[63]:


predict_my_movie=predict_movies("Titanic (1997)")


# In[64]:


predict_my_movie.head()


# In[ ]:




