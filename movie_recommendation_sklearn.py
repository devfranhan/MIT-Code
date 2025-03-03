
# Let's build a simple recommendation system using scikit-learn

# comparison:

# english run:
# (rcmm) franhan@Gabriels-MacBook-Pro MIT-Code % python3 recommendation.py
#   Recommended movie  Similarity score
# 0            Matrix          0.406838
# 1         Inception          0.375278
# 2      Pulp Fiction          0.295122
# 3     The Godfather          0.207020
# 4   Matrix Reloaded          0.203419

# portuguese run:
# (rcmm) franhan@Gabriels-MacBook-Pro MIT-Code % python3 recommendation.py
#    Recommended movie  Similarity score
# 0       Pulp Fiction          0.343776
# 1  O Poderoso Chefão          0.328244
# 2           A Origem          0.299667
# 3             Matrix          0.201008
# 4    Matrix Reloaded          0.041812

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def main():
    movies = {
        'title': ['Matrix', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Inception', 'Matrix Reloaded'],
        'genre': ['Action Sci-Fi', 'Crime Drama', 'Action Crime', 'Crime Drama', 'Action Sci-Fi', 'Action Sci-Fi'],
        'description': [
            'A computer hacker discovers the true nature of his reality and his role in the war against his controllers.',
            'An aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
            'When the threat known as the Joker emerges from his mysterious past, he wreaks havoc and chaos upon the people of Gotham.',
            'The lives of two hitmen, a boxer, a gangster\'s wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
            'A thief who steals corporate secrets through the use of dream-sharing technology is given the task of planting an idea into the mind of a CEO.',
            'Neo and the rebel leaders estimate that they have 72 hours until 250,000 probes discover Zion and destroy it along with its inhabitants.'
        ]
    }

    # testing portuguese cosine match
    # movies = {
    #     'title': ['Matrix', 'O Poderoso Chefão', 'O Cavaleiro das Trevas', 'Pulp Fiction', 'A Origem', 'Matrix Reloaded'],
    #     'genre': ['Ação Ficção Científica', 'Crime Drama', 'Ação Crime', 'Crime Drama', 'Ação Ficção Científica', 'Ação Ficção Científica'],
    #     'description': [
    #         'Um hacker descobre a verdadeira natureza de sua realidade e seu papel na guerra contra seus controladores.',
    #         'Um patriarca envelhecido de uma dinastia do crime organizado transfere o controle de seu império clandestino para seu filho relutante.',
    #         'Quando a ameaça conhecida como Coringa emerge de seu passado misterioso, ele espalha caos e destruição entre o povo de Gotham.',
    #         'As vidas de dois assassinos de aluguel, um boxeador, a esposa de um gângster e um casal de assaltantes de restaurante se entrelaçam em quatro histórias de violência e redenção.',
    #         'Um ladrão que rouba segredos corporativos através da tecnologia de compartilhamento de sonhos recebe a tarefa de implantar uma ideia na mente de um CEO.',
    #         'Neo e os líderes rebeldes estimam que têm 72 horas até que 250.000 sondas descubram Zion e a destruam junto com seus habitantes.'
    #     ]
    # }

    # load!
    df = pd.DataFrame(movies)

    # create unique column with genre and description
    df['movie_data'] = df['genre'] + ' ' + df['description']

    # vectorize!
    count_vectorizer = CountVectorizer()
    count_data = count_vectorizer.fit_transform(df['movie_data'])

    # similarity
    cosine_sim = cosine_similarity(count_data, count_data)


    def recommend_movies(title, qty, cosine_sim=cosine_sim):
        index = pd.Series(df.index, index=df['title']).drop_duplicates()
        idx = index[title]

        # score
        sim_scores = list(enumerate(cosine_sim[idx]))

        # order
        sim_scores_ordered = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # similar recommendations
        recommendations_qty = qty + 1
        recommendations = sim_scores_ordered[1:recommendations_qty]

        movies_indexes = [i[0] for i in recommendations]
        similarity_scores = [i[1] for i in recommendations]

        # display recommendation scores

        recommendations_df = pd.DataFrame({
            'Recommended movie': df['title'].iloc[movies_indexes].values,
            'Similarity score': similarity_scores
        })

        return recommendations_df

    recommended_movies = recommend_movies('The Dark Knight', 5)

    print(recommended_movies)

if __name__ == '__main__':
    main()

