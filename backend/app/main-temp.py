import asyncio
from datetime import datetime
import json
from typing import Dict
from fastapi import BackgroundTasks, FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import Null, and_, null, update
from sqlalchemy.orm import Session
# from models import User, Base
from schema import CreateFederatedLearning, ClientFederatedResponse, ClientModleIdResponse, ClientReceiveParameters
from sse_starlette import EventSourceResponse

from models.FederatedSession import FederatedSession, FederatedSessionClient
from helpers.federated_learning import start_federated_learning
from helpers.websocket import ConnectionManager
from utility.FederatedLearning import FederatedLearning
from db import get_db,engine
from models.User import User, Base
from schemas.UserSchema import RefreshToken, UserCreate, UserLogin, ClientSessionStatusSchema
from helpers.auth import create_refresh_token, decode_refresh_token, get_password_hash, verify_password, create_access_token, get_current_user
from dotenv import load_dotenv
from api.dataset_api import dataset_router
import os

from utility.user import get_unnotified_notifications
# from db import SessionLocal


load_dotenv()

# Create FastAPI app
app = FastAPI()

client1_url = os.getenv("CLIENT1_URL", "http://default-url.com")
origins = [ 
    "http://localhost:5173",
    "http://localhost:5174",
    client1_url  
]
print(origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Or specify allowed methods, e.g., ["GET", "POST"]
    allow_headers=["*"],  # Or specify allowed headers
)


# Create all tables
# Base.metadata.create_all(bind=engine)

websocket_manager = ConnectionManager()
        
# Signup route
@app.post("/signup", status_code=201)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # if db.query(User).filter(User.email == user.email).first():
    #     raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    new_user = User(
        username=user.username,
        data_url=user.data_url,
        hashed_password=get_password_hash(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}

# Login route
@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    def is_invalid(db_user: User):
        not verify_password(user.password, db_user.hashed_password)
        
    return create_tokens(
        db,
        user.username,
        HTTPException(status_code=400, detail="Invalid credentials"),
        is_invalid
    )

@app.post("/refresh-token")
def refresh_token(token: RefreshToken, db: Session = Depends(get_db)):
    try:
        # Decode the refresh token
        payload = decode_refresh_token(token.refresh_token)
        username: str = payload.get("sub")
        
        expiry = datetime.fromtimestamp(payload.get('exp'))
        
        if(expiry > datetime.now()):
            if username is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            def validate_user(db_user: User):
                return db_user.refresh_token != token.refresh_token

            return create_tokens(
                db,
                username,
                HTTPException(status_code=401, detail="Invalid token"),
                validate_user
            )
        else:
            raise HTTPException(status_code=440, detail="Session Timed Out!")
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

    
def create_tokens(db: Session, username: str, exception: HTTPException, check = null):
    # Verify if the refresh token exists in the database
    db_user = db.query(User).filter(User.username == username).first()
    if (not db_user) or (check and check(db_user)):
        raise exception
    
    # Create a new access token
    new_access_token = create_access_token(data={"sub": db_user.username})
    new_refresh_token = create_refresh_token(data={"sub": db_user.username})
    
    # Store refresh token in database
    db_user.refresh_token = new_refresh_token
    db.commit()

    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }


@app.post("/logout")
def logout(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    current_user.refresh_token = None
    db.commit()
    return {"msg": "User logged out"}

# Protected API route
@app.get("/users/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username}

@app.get("/client/initiated_sessions", response_model=list[ClientSessionStatusSchema])
def get_initiated_jobs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(
            FederatedSession.id,
            FederatedSession.curr_round,
            FederatedSession.max_round,
            FederatedSession.session_price,
            FederatedSession.training_status,
            FederatedSessionClient.status.label('client_status')
        ).outerjoin(
            FederatedSessionClient,
            (FederatedSession.id == FederatedSessionClient.session_id) &
            (FederatedSessionClient.user_id == current_user.id)
        ).filter(FederatedSession.admin_id == current_user.id).all()
    
    return [ClientSessionStatusSchema.model_validate(dict(zip(
            ["session_id","curr_round", "max_round", "session_price", "training_status", "client_status"], session
        ))) for session in sessions]

@app.get("/client/participated_sessions", response_model = list[ClientSessionStatusSchema])
def get_participated_sessions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(
        FederatedSession.curr_round,
        FederatedSession.max_round,
        FederatedSession.session_price,
        FederatedSession.training_status,
        FederatedSessionClient.status.label("client_status")
    ).join(
        FederatedSessionClient, FederatedSession.id == FederatedSessionClient.session_id
    ).filter(
        FederatedSessionClient.user_id == current_user.id
    ).all()
    return [ClientSessionStatusSchema.model_validate(dict(zip(
            ["curr_round", "max_round", "session_price", "training_status", "client_status"], session
        ))) for session in sessions]
    


@app.get("/notifications/stream")
async def notifications_stream(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    async def event_generator():
        while True:
            # Check if client has disconnected
            if await request.is_disconnected():
                break
            
            # # Expire all cached objects to force fresh queries
            # db.expire_all()
        
            # Fetch notifications for the current user
            user_notifications = get_unnotified_notifications(user = current_user, db = db)
            
            if(len(user_notifications) > 0):
                data = [n.message for n in user_notifications]

                # Send the data as an SSE event
                yield {
                    "event": "new_notifications",   
                    "data": json.dumps(data),
                }
                

                for notification in user_notifications:
                    notification.notified_at = datetime.now()
                db.commit()

            # Wait for 5 seconds before sending the next update
            await asyncio.sleep(5)

    return EventSourceResponse(event_generator())


federated_manager = FederatedLearning()

@app.post("/create-federated-session")
async def create_federated_session(
    federated_details: CreateFederatedLearning,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):  
    print(federated_details.fed_info)
    session: FederatedSession = federated_manager.create_federated_session(current_user, federated_details.fed_info, request.client.host)
    
    # await websocket_manager.broadcast({
    #     'type': "new-session",
    #     'message': "New Federated Session Avaliable!"
    # })
    
    # message = {
    #     'type': "new-session",
    #     'message': "New Federated Session Avaliable!",
    #     'session_id': session.id
    # }
    
    # add_notifications_for_recently_active_users(db=db, message=message, valid_until=session.wait_till, excluded_users=[current_user])
    
    try:
        background_tasks.add_task(start_federated_learning, federated_manager, current_user, session, db)
        print("Background Task Added")
    except Exception as e:
        print(f"An error occurred while adding background process {Exception}")
        return {"message": "An error occurred while starting federated learning."}
    
    return {
        "message": "Federated Session has been created!",
        "session_id": session.id
    }


@app.get('/get-all-federated-sessions')
def get_all_federated_session(current_user: User = Depends(get_current_user)):
    return [
        {
            'id': id,
            'training_status': training_status,
            'name': federated_info.get('organisation_name')
        }
        for [id, training_status, federated_info]
        in federated_manager.get_my_sessions(current_user)
    ]

@app.get('/get-federated-session/{session_id}')
def get_federated_session(session_id: int, current_user: User = Depends(get_current_user)):
    try:
        federated_session_data = federated_manager.get_session(session_id)
        client = next((client for client in federated_session_data.clients if client.user_id == current_user.id), None)

        federated_response = {
            'federated_info': federated_session_data.federated_info,
            'training_status': federated_session_data.training_status,
            'client_status': client.status if client else 1,
            'session_price': federated_session_data.session_price
        }

        return federated_response
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    

@app.post('/submit-client-price-response')
def submit_client_price_response(client_response: ClientFederatedResponse, request: Request, db: Session = Depends(get_db)):
    '''
        decision : 1 means client accepts the price, -1 means client rejects the price
        training_status = 2 means the training process should start
    '''
    try:
        session_id = client_response.session_id
        decision = client_response.decision
        
        session = federated_manager.get_session(session_id)
        if(session):
            # Fetch the FederatedSession by session_id
            federated_session = db.query(FederatedSession).filter_by(id = session_id).first()
            if not federated_session:
                raise HTTPException(status_code=404, detail="Federated session not found")
            # Update training_status based on the decision
            if decision == 1:
                federated_session.training_status = 2  # Update training_status to 2 (start training)
            elif decision == -1:
                federated_session.training_status = -1  # Keep or set to a default status for rejection
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid decision value. Must be 1 (accept) or -1 (reject)."
                )
            # Commit changes to the database
            db.commit()
            

            return {'success': True, 'message': 'Training status updated successfully'}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post('/submit-client-federated-response')
def submit_client_federated_response(client_response: ClientFederatedResponse, request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    '''
        decision : 1 means client accepts and 0 means rejects
        client_status = 2 means client has accepted the request
        client_status = 3 means client rejected the request
    '''
    session_id = client_response.session_id
    decision = client_response.decision
    client_status = 3
    if decision == 1:
        client_status = 2
    
    session = federated_manager.get_session(session_id)
    
    if(session):
        client = db.query(FederatedSessionClient).filter_by(session_id = session_id, user_id = current_user.id).first()
        if not client:
            federated_session_client = FederatedSessionClient(
                user_id = current_user.id,
                session_id = session_id,
                status = client_status,
                ip = request.client.host
            )
            
            db.add(federated_session_client)
        else:
            client.status = client_status
        db.commit()
    
    return { 'success': True, 'message': 'Client Decision has been saved'}


@app.post('/update-client-status-four')
def update_client_status_four(request: ClientModleIdResponse, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    '''
        Client have received the model parameters and waiting for server to start training
    '''

    session_id = request.session_id
    local_model_id = request.local_model_id
    db.execute(
        update(FederatedSessionClient)
        .where(and_(
            FederatedSessionClient.user_id == current_user.id,
            FederatedSessionClient.session_id == session_id
        ))
        .values(
            status = 4,
            local_model_id = local_model_id
        )
    )
    
    db.commit()

    return {'message': 'Client Status Updated to 4'}

@app.get('/get-model-parameters/{session_id}')
def get_model_parameters(session_id: str):
    '''
        Client have received the model parameters and waiting for server to start training
    '''
    global_parameters = json.loads(federated_manager.get_session(session_id).global_parameters)
    
    response_data = {
        "global_parameters": global_parameters,
        "is_first": 0
    }

    # Save global_parameters string into a file
    file_path = "global_parameters.txt"  # Specify the desired file path and name
    with open(file_path, "a") as file:
        file.write("\n---\n")  # Add a separator before each new entry
        file.write(json.dumps(global_parameters))  # Append the JSON string
        file.write("\n")  # Add a newline after the entry for readability
    print(f"Global parameters have been saved to {file_path}.")
    return response_data

@app.post('/receive-client-parameters')
def receive_client_parameters(request: ClientReceiveParameters,  current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session_id = request.session_id
    client_parameter = request.client_parameter
    
    session_data = db.query(FederatedSession).filter(FederatedSession.id == session_id).first()
    
    if not session_data:
        raise HTTPException(status_code=404, detail=f"Federated Session with ID {session_id} not found!")
    
    # Deserialize client_parameters from JSON to a Python dictionary
    existing_parameters = json.loads(session_data.client_parameters) if session_data.client_parameters else {}
    
    existing_parameters[str(current_user.id)] = client_parameter
    session_data.client_parameters = json.dumps(existing_parameters)
    
    db.commit()
    
    # federated_manager.federated_sessions[session_id]['client_parameters'][client_id] = request.client_parameter
    return {"message": "Client Parameters Received"}

@app.get('/get-all-completed-trainings')
def get_training_results():
    # iterate ove Global_test_results folder and return the completed sessions' results
    try:
        results_dir = "Global_test_results"
        results = []
        for file in os.listdir(results_dir):
            if file.endswith(".json"):
                with open(os.path.join(results_dir, file), "r") as f:
                    result = json.load(f)
                    # return only session_id and organisation_name
                    # only save session_id from file name not all filename
                    results.append({
                        "session_id": file.split("_")[0],
                        "org_name": result["session_data"]["organisation_name"]
                    })
        return {"results": results}

    except Exception as e:
        return {"message": f"No training results"}


@app.get('/get-training-result/{session_id}')
def get_training_results(session_id: str):
    # iterate ove Global_test_results folder and return the session_id's results
    try:
        results_dir = "Global_test_results"
        with open(os.path.join(results_dir, f"{session_id}_test_results.json"), "r") as f:
            result = json.load(f)
            return result
    except Exception as e:
        return {"message": f"No training results with this session_id"}


app.include_router(dataset_router,tags=["Dataset"])
