from typing import List
import databases
import sqlalchemy
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import os
import urllib


host_server = os.environ.get("host_server", "localhost")
db_server_port = urllib.parse.quote_plus(str(os.environ.get("db_server_port", "5432")))
database_name = os.environ.get("database_name", "fastapi")
db_username = urllib.parse.quote_plus(str(os.environ.get("db_username", "postgres")))
db_password = urllib.parse.quote_plus(str(os.environ.get("db_password", "secret")))
ssl_mode = urllib.parse.quote_plus(str(os.environ.get("ssl_mode", "prefer")))
DATABASE_URL = "postgresql://{}:{}@{}:{}/{}?sslmode={}".format(
    db_username, db_password, host_server, db_server_port, database_name, ssl_mode
)


database = databases.Database(DATABASE_URL)

metadata = sqlalchemy.MetaData()

scores = sqlalchemy.Table(
    "scores",
    metadata,
    sqlalchemy.Column("patient_mrn", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("risk_score", sqlalchemy.Float),
    sqlalchemy.Column("update_date", sqlalchemy.String),
)

engine = sqlalchemy.create_engine(DATABASE_URL, pool_size=3, max_overflow=0)

metadata.create_all(engine)


class Patient(BaseModel):
    patient_mrn: List[int]


class ScoreIn(BaseModel):
    patient_mrn: int
    risk_score: float
    update_date: str


class Score(BaseModel):
    patient_mrn: int
    risk_score: float
    update_date: str


app = FastAPI(title="Readmissions REST API using FastAPI PostgreSQL Async EndPoints")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware)


@app.on_event("startup")
async def startup():
    await database.connect()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


@app.post("/scores/", response_model=Score, status_code=status.HTTP_201_CREATED)
async def create_score(score: ScoreIn):
    query = scores.insert().values(
        patient_mrn=score.patient_mrn,
        risk_score=score.risk_score,
        update_date=score.update_date,
    )
    last_record_id = await database.execute(query)
    return {**score.dict()}


@app.get("/scores/", response_model=List[Score], status_code=status.HTTP_200_OK)
async def read_all_scores(skip: int = 0, take: int = 20):
    query = scores.select().offset(skip).limit(take)
    return await database.fetch_all(query)


@app.get(
    "/scores/{score_date}/", response_model=List[Score], status_code=status.HTTP_200_OK
)
async def read_scores_by_date(score_date: str):
    query = scores.select().where(scores.c.update_date == score_date)
    return await database.fetch_all(query)


@app.post("/predict/", response_model=List[Score], status_code=status.HTTP_201_CREATED)
async def predict_score(patients: Patient):
    query = scores.select().where(
        scores.c.patient_mrn.in_([item for item in patients.patient_mrn])
    )
    return await database.fetch_all(query)
