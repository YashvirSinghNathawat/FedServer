from fastapi import APIRouter, Depends, HTTPException
from models.Dataset import Dataset, DatasetFile
from models.Benchmark import Benchmark
from fastapi import APIRouter
from sqlalchemy.orm import Session, joinedload
from db import get_db
from schemas.Dataset_Schema import DatasetCreate, BenchmarkCreate, BenchmarkResponse, DatasetUpdateRequest
from typing import List, Optional

dataset_router = APIRouter()


# Dataset Endpoints
@dataset_router.post("/datasets/add-dataset/")
def create_dataset(dataset: DatasetCreate, db: Session = Depends(get_db)):
    # Check if the dataset name already exists
    existing_dataset = db.query(Dataset).filter(Dataset.code == dataset.code).first()
    
    if existing_dataset:
        raise HTTPException(status_code=400, detail=f'Dataset with this code {dataset.code} already exists')
    
    # Create a new Dataset instance
    new_dataset = Dataset(
        name=dataset.name,
        code=dataset.code,
        description=dataset.description,
        source=dataset.source,
    )
    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)
    return {
            'message':'Dataset has been added!'}

@dataset_router.get("/datasets/{code}")
def get_dataset(code: str, db: Session=Depends(get_db)):
    dataset = db.query(Dataset).options(joinedload(Dataset.benchmarks),
                                        joinedload(Dataset.files)).filter(Dataset.code == code).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with code {code} does not exists")
    return dataset

@dataset_router.get("/datasets/")
def get_all_dataset(db: Session=Depends(get_db)):
    datasets = db.query(Dataset).options(joinedload(Dataset.benchmarks)).all()
    # Format the response to include benchmark count
    response = [
        {
            "id": dataset.id,
            "name": dataset.name,
            "code": dataset.code,
            "description": dataset.description,
            "source": dataset.source,
            "benchmark_count": len(dataset.benchmarks)  # Count benchmarks directly from the relationship
        }
        for dataset in datasets
    ]
    return response

@dataset_router.put("/datasets/{code}")
def update_dataset(code: str, update_request: DatasetUpdateRequest,db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.code == code).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with code {code} does not exists")
    
    # Update the dataset's fields based on the request body
    if update_request.name is not None:
        dataset.name = update_request.name
    if update_request.description is not None:
        dataset.description = update_request.description
    if update_request.source is not None:
        dataset.source = update_request.source
    if update_request.columns is not None:
        dataset.columns = update_request.columns
    if update_request.files is not None:
        dataset.files = [
            DatasetFile(
                name = file.name,
                hdfs_path = file.hdfs_path,
                is_folder = file.is_folder,
                description = file.description
            )
            for file in update_request.files
        ]
        
    db.commit()
    db.refresh(dataset)
    
    return dataset


@dataset_router.post('/benchmarks/add-benchmarks',response_model=dict)
def create_benchmark(benchmark: BenchmarkCreate,db: Session = Depends(get_db)):
     # Check if the benchmark already exists
    existing_benchmark = db.query(Benchmark).filter(
        Benchmark.task == benchmark.task,
        Benchmark.dataset_id == benchmark.dataset_id,
        Benchmark.model_name == benchmark.model_name,
        Benchmark.benchmark_metric == benchmark.benchmark_metric
    ).first()

    if existing_benchmark:
        raise HTTPException(
            status_code=400,
            detail="Benchmark with the same task, dataset, model, and metric already exists."
        )
    db_benchmark = Benchmark(
        task = benchmark.task,
        dataset_id=benchmark.dataset_id,
        model_name=benchmark.model_name,
        benchmark_metric=benchmark.benchmark_metric,
        metrics=benchmark.metrics
    )
    
    try:
        db.add(db_benchmark)
        db.commit()
        db.refresh(db_benchmark)
    except Exception as e:
        db.roll_back()
        raise HTTPException(status_code=400, detail=f"Error creating benchmark: {e}")
    
    # Return a success message
    return {"id": db_benchmark.id, "message": "Benchmark created successfully"}
        

@dataset_router.get("/benchmarks/",response_model=List[BenchmarkResponse])
def get_all_benchmarks(
    dataset_id: Optional[int] = None,
    benchmark_id: Optional[int] = None,
    db: Session=Depends(get_db)
):
    if benchmark_id is not None:
        benchmarks = db.query(Benchmark).filter(Benchmark.id == benchmark_id).all()
    elif dataset_id is not None:
        benchmarks = db.query(Benchmark).filter(Benchmark.dataset_id == dataset_id).all()
    else:
        benchmarks = db.query(Benchmark).all()
    return benchmarks

@dataset_router.get("/benchmarks/{benchmark_id}")
def get_benchmark(benchmark_id: int, db: Session = Depends(get_db)):
    benchmark = db.query(Benchmark).filter(Benchmark.id == benchmark_id).first()
    if not benchmark:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return benchmark

@dataset_router.delete("/benchmarks/delete-benchmark/{benchmark_id}")
def delete_benchmark(benchmark_id: int, db: Session = Depends(get_db)):
    benchmark = db.query(Benchmark).filter(Benchmark.id == benchmark_id).first()

    if not benchmark:
        raise HTTPException(status_code=404, detail="Benchmark not found.")

    try:
        db.delete(benchmark)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error deleting benchmark: {e}")

    return {"message": "Benchmark deleted successfully"}




