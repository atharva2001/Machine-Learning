import streamlit as st # type: ignore
import pandas as pd # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore
import numpy as np
from scipy.sparse import csr_matrix

# Load data
jobs = pd.read_csv("jobs.csv")
users = pd.read_csv("users.csv")
applies = pd.read_csv("applications.csv")

# --------- Collaborative Filtering (Job-Job Similarity) ---------
def create_matrix(df):
    N = len(df['user_id'].unique())
    M = len(df['job_id'].unique())

    user_mapper = dict(zip(np.unique(df["user_id"]), list(range(N))))
    job_mapper = dict(zip(np.unique(df["job_id"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["user_id"])))
    job_inv_mapper = dict(zip(list(range(M)), np.unique(df["job_id"])))

    user_index = [user_mapper[i] for i in df['user_id']]
    job_index = [job_mapper[i] for i in df['job_id']]

    X = csr_matrix((np.ones(len(user_index)), (user_index, job_index)), shape=(N, M))
    return X, user_mapper, job_mapper, user_inv_mapper, job_inv_mapper

X, user_mapper, job_mapper, user_inv_mapper, job_inv_mapper = create_matrix(applies)

def find_similar_jobs(job_id, X, k, job_mapper, job_inv_mapper):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(X.T)
    job_index = job_mapper[job_id]
    distances, indices = model.kneighbors(X.T[job_index], n_neighbors=k+1)

    similar_jobs = []
    for i in range(1, len(distances.flatten())):
        similar_jobs.append((job_inv_mapper[indices.flatten()[i]], distances.flatten()[i]))
    return similar_jobs

def recommend_jobs_cf(user_id, X, user_mapper, job_mapper, job_inv_mapper, n_recommendations=5):
    if user_id not in user_mapper:
        return pd.DataFrame()

    user_index = user_mapper[user_id]
    user_applied_jobs = X[user_index].nonzero()[1]

    all_recommendations = {}
    for job_index in user_applied_jobs:
        job_id = job_inv_mapper[job_index]
        similar_jobs = find_similar_jobs(job_id, X, k=n_recommendations, job_mapper=job_mapper, job_inv_mapper=job_inv_mapper)
        for sim_job_id, score in similar_jobs:
            if sim_job_id not in user_applied_jobs and sim_job_id not in all_recommendations:
                all_recommendations[sim_job_id] = score

    sorted_recommendations = sorted(all_recommendations.items(), key=lambda x: x[1])[:n_recommendations]
    results = []
    for job_id, score in sorted_recommendations:
        job_info = jobs[jobs['job_id'] == job_id].iloc[0]
        results.append({
            "job_id": job_id,
            "skills": job_info["skills"],
            "description": job_info["description"],
            "similarity": f"{score:.4f}"
        })
    return pd.DataFrame(results)


# --------- Content-Based Filtering ---------
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(","))
job_skill_matrix = vectorizer.fit_transform(jobs["skills"])

def recommend_jobs_cb_from_userid(user_id, top_k=5):
    user_row = users[users["user_id"] == user_id].iloc[0]
    user_skills = user_row["skills"]
    return recommend_jobs_cb(user_skills, top_k)

def recommend_jobs_cb(user_skills, top_k=5):
    user_vec = vectorizer.transform([user_skills])
    similarity_scores = cosine_similarity(user_vec, job_skill_matrix).flatten()

    job_scores = pd.DataFrame({
        "job_id": jobs["job_id"],
        "score": similarity_scores
    })

    job_scores = job_scores.merge(jobs[["job_id", "skills", "description"]], on="job_id")
    recommendations = job_scores.sort_values("score", ascending=False).head(top_k)
    return recommendations


# --------- Streamlit UI ---------
st.title("Job Recommendation System")

option = st.sidebar.radio(
    "Select Recommendation Method:",
    ["Collaborative Filtering (by User ID)", 
     "Content-Based (by User ID Skills)", 
     "Content-Based (by Typed Skills)"]
)

if option == "Collaborative Filtering (by User ID)":
    user_id = st.number_input("Enter User ID:", min_value=1, max_value=users["user_id"].max())
    if st.button("Recommend"):
        recs = recommend_jobs_cf(int(user_id), X, user_mapper, job_mapper, job_inv_mapper, n_recommendations=5)
        st.write("### Recommended Jobs (Collaborative Filtering)")
        st.dataframe(recs)

elif option == "Content-Based (by User ID Skills)":
    user_id = st.number_input("Enter User ID:", min_value=1, max_value=users["user_id"].max(), key="cb_user")
    user_row = users[users["user_id"] == user_id].iloc[0]
    user_skills = user_row["skills"]
    if st.button("Recommend", key="cb_btn"):
        recs = recommend_jobs_cb_from_userid(int(user_id), top_k=5)
        st.write(f"### Recommended Jobs (Content-Based, from User Skills)\nUser Skills: {user_skills}")
        st.dataframe(recs)

else:  # Content-Based by Typed Skills
    user_skills = st.text_input("Enter your skills (comma-separated):", "Python,SQL,Machine Learning")
    if st.button("Recommend", key="typed_btn"):
        recs = recommend_jobs_cb(user_skills, top_k=5)
        st.write("### Recommended Jobs (Content-Based, from Typed Skills)")
        st.dataframe(recs)
