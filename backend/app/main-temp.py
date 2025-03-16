from fastapi import FastAPI, BackgroundTasks, Request, HTTPException, WebSocket, WebSocketDisconnect
from schema import FederatedLearningInfo, User, Parameter, CreateFederatedLearning, ClientFederatedResponse, \
    ClientReceiveParameters
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from utility.FederatedLearning import FederatedLearning
from utility.ConnectManager import ConnectionManager
from utility.Server import Server
import asyncio
import json
import asyncio
import uuid
from utility.test import Test
import os

'''
    Naming Conventions as per PEP- https://peps.python.org/pep-0008/#function-and-variable-names
    Classes - CapWords Convention
    Methods - Function names should be lowercase, with words separated by underscores as necessary to improve readability.
    Variables - Variable names follow the same convention as function names.
'''


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

event = asyncio.Event()

clients_data = []


class MessageType:
    GET_MODEL_PARAMETERS_START_BACKGROUND_PROCESS = "get_model_parameters_start_background_process"
    START_TRAINING = "start_training"


# Utility Function
def generate_unique_token():
    token = str(uuid.uuid4())
    return token


def generate_session_token():
    session_token = generate_unique_token()
    # Check if token is already assigned to other federated session
    return session_token


def generate_user_token():
    user_token = generate_unique_token()
    # Check if token is already assigned to other federated session
    return user_token


# Usage
federated_manager = FederatedLearning()
connection_manager = ConnectionManager()


async def wait_for_client_confirmation(session_id: str):
    session_data = federated_manager.federated_sessions[session_id]
    all_ready_for_training = False

    while not all_ready_for_training:
        await asyncio.sleep(5)
        all_ready_for_training = all(client['status'] != 1 for client in session_data['clients_status'].values())
        print("Waiting for client confirmations....Stage 1")

    print("All Clients have taken their decision.")


async def send_message_with_type(client_id: str, message_type: str, data: dict, session_id: str):
    message = {
        "type": message_type,
        "data": data,
        "session_id": session_id
    }
    json_message = json.dumps(message)
    print("json model sent before the training signal: ", json_message)
    await connection_manager.send_message(json_message, client_id)


async def send_model_configuration(client_id: str, session_id: str):
    model_data = federated_manager.federated_sessions[session_id]
    model_config = model_data['federated_info']
    model_config_dict = model_config.dict()  # Convert to dictionary
    await send_message_with_type(client_id, MessageType.GET_MODEL_PARAMETERS_START_BACKGROUND_PROCESS,
                                 model_config_dict, session_id)


async def wait_for_all_clients_to_stage_four(session_id: str):
    # Implement the logic to wait for all clients to confirm that they have started background process
    session_data = federated_manager.federated_sessions[session_id]
    interested_clients = session_data['interested_clients']

    while True:
        all_clients_ready = True
        for client_id in interested_clients:
            if session_data['clients_status'][client_id]['status'] != 4:
                all_clients_ready = False
                break

        if all_clients_ready:
            print("All clients have reached stage four.")
            break
        else:
            await asyncio.sleep(5)
            print("Waiting for all clients to reach stage four.")


async def send_model_configs_and_wait_for_confirmation(session_id: str):
    interested_clients = federated_manager.federated_sessions[session_id]['interested_clients']
    for client in interested_clients:
        await send_model_configuration(client, session_id)

    # Wait for all clients to confirm they have started their background process
    await wait_for_all_clients_to_stage_four(session_id)


async def send_training_signal_to_clients(session_id: str):
    session_data = federated_manager.federated_sessions[session_id]
    interested_clients = session_data['interested_clients']
    data = {
        'session_id': session_id  #dummy data, not accesses at client side
    }
    print("Before Sending Signal : ")
    for client_id in interested_clients:
        print(client_id)
        await send_message_with_type(client_id, MessageType.START_TRAINING, data,
                                     session_id)  #session_id is dummy data here
    print("Training Signal Sent to all clients")


async def send_training_signal_and_wait_for_clients_training(session_id: str):
    await send_training_signal_to_clients(session_id)

    session_data = federated_manager.federated_sessions[session_id]
    num_interested_clients = len(session_data['interested_clients'])
    print("Error Check : ", len(session_data['client_parameters']), num_interested_clients)
    while len(session_data['client_parameters']) < num_interested_clients:
        await asyncio.sleep(5)
        print(f"Waiting for {num_interested_clients - len(session_data['client_parameters'])}")

    # print("All clients have sent parameters. Starting Aggregation...", session_data['client_parameters'])


async def start_federated_learning(session_id: str):
    """
    Background task to manage federated learning rounds.

    This function runs in the background, waiting for client responses before proceeding with each round
    of federated learning.

    Each round consists of:
    1. Setting the current round number (`curr_round`) in the server.
    2. Printing round information.
    3. Receiving parameters from clients.
    4. Aggregating weights using federated averaging with neural networks.

    """
    try:
        session_data = federated_manager.federated_sessions[
            session_id]  # session_data points to same object variables in Python that point to mutable objects (like dictionaries) actually reference the same underlying object in memory.

        # Wait for client confirmation of interest
        await wait_for_client_confirmation(session_id)

        # Send Model Configurations to interested clients and wait for their confirmation
        await send_model_configs_and_wait_for_confirmation(session_id)

        #############################################
        # code used to get instance of testing unit
        # Here Input has to be taken in future for the metrics
        test = Test(session_id, session_data)
        #############################################

        # Start Training
        for i in range(1, session_data['max_round'] + 1):
            print("-" * 50)
            federated_manager.federated_sessions[session_id]['curr_round'] = i
            print(f"Round {i}")
            print("-" * 50)

            await send_training_signal_and_wait_for_clients_training(session_id)
            # Aggregate
            print("Done upto just before aggregation...")
            federated_manager.aggregate_weights_fedAvg_Neural(session_id)

            ################## Testing start
            results = test.start_test(federated_manager.federated_sessions[session_id]['global_parameters'])
            print("Global test results: ", results)
            ################## Testing end

        # Save test results for future reference
        """ Yashvir: here you can delete the data of this session from the federated_sessions dictionary after saving the results 
            , saved results contains session_data and test_results across all rounds
        """
        test.save_test_results()
    except Exception as e:
        print(f"Error in Starting Background Process: {e}")


def add_interested_user_to_session(client_token, session_token: str, request: Request, admin):
    """
    Generates a token for user which will be used to validate the sender, and the token will be bound to a spefic session.
    :param client_token: This will be authentication token for a user in future when establish authentication each user has a unique user_id
    :param session_token: This will be unique token for each session
    :param admin: This user is admin of this request or not
    :param request:
    :return:
    """
    federated_manager.federated_sessions[session_token]['clients_status'][client_token]['status'] = 2
    federated_manager.federated_sessions[session_token]["interested_clients"][client_token] = {
        "ip": request.client.host,
    }
    if admin:
        federated_manager.federated_sessions[session_token]["admin"] = client_token


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await connection_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "pong":
                print(f"Received pong from client {client_id}")
                continue
            print(f"Received message: {message}")

    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
        print(f"Client {client_id} disconnected")
    except Exception as e:
        print(f"An error occurred: {str(e)}")




@app.post('/sign-in')
def signIn(request: User):
    user_token = generate_user_token()
    clients_data.append(user_token)
    print(f"{request.name} is registered")
    return {"message": "Client Registered Successfully", "client_token": user_token}


@app.post("/create-federated-session")
def create_federated_session(federated_details: CreateFederatedLearning, request: Request,
                             background_tasks: BackgroundTasks):
    session_token = generate_session_token()
    federated_manager.create_federated_session(session_token, federated_details.fed_info, clients_data)

    add_interested_user_to_session(federated_details.client_token, session_token, request, admin=True)
    try:
        background_tasks.add_task(start_federated_learning, session_token)
        print("Background Task Added")
    except Exception as e:
        print(f"An error occurred while adding background process {Exception}")
        return {"message": "An error occurred while starting federated learning."}

    return {"message": "Federated Session has been created!"}


@app.get('/get-all-federated-sessions')
def get_all_federated_session():
    federated_session = []
    for session_id, session_data in federated_manager.federated_sessions.items():
        federated_session.append({
            'session_id': session_id,
            "training_status": session_data["training_status"]
        })
    return {"federated_session": federated_session}


@app.get('/get-federated-session/{session_id}')
def get_federated_session(session_id: str, client_id: str):
    first_session_key = list(federated_manager.federated_sessions.keys())[0]
    try:
        federated_session_data = federated_manager.federated_sessions[session_id]
        federated_response = {
            'federated_info': federated_session_data['federated_info'],
            'training_status': federated_session_data['training_status'],
            'client_status': federated_session_data['clients_status'][client_id]['status']
        }
        return federated_response
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post('/submit-client-federated-response')
def submit_client_federated_response(client_response: ClientFederatedResponse, request: Request):
    '''
        decision : 1 means client accepts and 0 means rejects
        client_status = 2 means client has accepted the request
        client_status = 3 means client rejected the request
    '''
    try:
        session_id = client_response.session_id
        client_id = client_response.client_id
        decision = client_response.decision
        if decision == 1:
            federated_manager.federated_sessions[session_id]['clients_status'][client_id]['status'] = 2
            add_interested_user_to_session(client_id, session_id, request, admin=False)
        else:
            federated_manager.federated_sessions[session_id]['clients_status'][client_id]['status'] = 3
        return {'message': 'Client Decision has been saved'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post('/update-client-status-four')
def update_client_status_four(request: ClientFederatedResponse):
    '''
        Client have received the model parameters and waiting for server to start training
    '''
    client_id = request.client_id
    session_id = request.session_id
    federated_manager.federated_sessions[session_id]['clients_status'][client_id]['status'] = 4
    return {'message': 'Client Status Updated to 4'}


@app.get('/get-model-parameters/{session_id}')
def get_model_parameters(session_id: str):
    '''
        Client have received the model parameters and waiting for server to start training
    '''
    global_parameters = federated_manager.federated_sessions[session_id]['global_parameters']
    response_data = {
        "global_parameters": global_parameters,
        "is_first": 0
    }
    if len(global_parameters) == 0:
        response_data['is_first'] = 1
    return response_data


@app.post('/receive-client-parameters')
def receive_client_parameters(request: ClientReceiveParameters):
    session_id = request.session_id
    client_id = request.client_id
    federated_manager.federated_sessions[session_id]['client_parameters'][client_id] = request.client_parameter
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


@app.get('/test')
def test_server():
    return {"message": "Server is started..."}