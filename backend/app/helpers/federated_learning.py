
from datetime import datetime
from typing import Dict
from requests import session
from sqlalchemy import Null
from helpers.websocket import ConnectionManager
from models.FederatedSession import FederatedSession, FederatedSessionClient
from utility import FederatedLearning
from models import User
import asyncio
import json
from db import engine
from sqlalchemy.orm import Session
from models.Notification import Notification
from utility.test import Test
from models.Benchmark import Benchmark
from utility.notification import add_notifications_for, add_notifications_for_user, add_notifications_for_recently_active_users
from utility.SampleSizeEstimation import calculate_required_data_points



def fetch_benchmark_and_calculate_price(session_data: FederatedSession, db: Session)->float :
    benchmark = db.query(Benchmark).filter(
            Benchmark.id == 1
        ).first()
    if not benchmark:
        raise ValueError("Benchmark not found!")
    
    # Extract benchmark mean and standard deviation
    metric_name = benchmark.benchmark_metric
    metrics = benchmark.metrics
    baseline_mean = metrics[metric_name]["mean"]
    baseline_std = metrics[metric_name]["std_dev"]
    
    new_mean = session_data.federated_info.get("std_mean")
    new_std = session_data.federated_info.get("std_deviation")
    
    if new_mean is None or new_std is None:
        raise ValueError("New model metrics (std_mean and std_deviation) are missing in session data.")
    
    # Extract num_predictors strictly from input_shape
    input_shape = session_data.federated_info["model_info"].get("input_shape")

    if not input_shape:
        raise ValueError("Input shape is missing in model_info.")

    try:
        shape_tuple = eval(input_shape)  # Convert string "(128,128,1)" → tuple (128,128,1)
        num_predictors = 1
        for dim in shape_tuple:
            num_predictors *= dim  # Compute total number of input features
    except Exception:
        raise ValueError(f"Invalid input_shape format: {input_shape}")
    # Calculate the required data points (price)
    price = calculate_required_data_points(
        baseline_mean, baseline_std, new_mean, new_std, num_predictors
    )
    return price
    
async def start_federated_learning(federated_manager: FederatedLearning, user: User, session_data: FederatedSession, db: Session):
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
    
    # Fetch benchmark stats and calculate required data points (price)
    required_data_points = fetch_benchmark_and_calculate_price(session_data, db)

    print("Price for Training :", required_data_points)
    # Store the calculated price in the session
    federated_session = db.query(FederatedSession).filter_by(id=session_data.id).first()
    if federated_session:    
        federated_session.session_price = required_data_points
        db.commit()
        db.refresh(federated_session)
    else:
        raise ValueError(f"FederatedSession with ID {session_data.id} not found.")
    
    # Send the price to the client and wait for approval
    approved = await wait_for_price_confirmation(federated_manager, session_data.id)

    if not approved:
        print(f"Client {user.id} declined the price. Aborting training session {session_data.id}.")
        return  # Stop execution if client rejects price

    print(f"Client {user.id} accepted the price. Starting federated learning session {session_data.id}.")
    
    # Send the price to the client and wait for approval
    await wait_for_price_confirmation(federated_manager, session_data.id)
    
    message = {
        'type': "new-session",
        'message': "New Federated Session Avaliable!",
        'session_id': session_data.id
    }
    
    add_notifications_for_recently_active_users(db=db, message=message, valid_until=session_data.wait_till, excluded_users=[user])
    
    # Wait for client confirmation of interest
    await wait_for_client_confirmation(federated_manager, session_data.id)

    # Send Model Configurations to interested clients and wait for their confirmation
    await send_model_configs_and_wait_for_confirmation(federated_manager, session_data.id)

    #############################################
    # code used to get instance of testing unit
    # Here Input has to be taken in future for the metrics
    test = Test(session_data.id, session_data)
    #############################################

    # Start Training
    for i in range(1, session_data.max_round + 1):
        print("-" * 50)
        federated_session = db.query(FederatedSession).filter_by(id = session_data.id).first()
        federated_session.curr_round = i
        print(f"Round {i}")
        print("-" * 50)

        await send_training_signal_and_wait_for_clients_training(federated_manager, session_data.id)
        # Aggregate
        print("Done upto just before aggregation...")
        federated_manager.aggregate_weights_fedAvg_Neural(session_data.id)
        with Session(engine) as db:
            federated_session = db.query(FederatedSession).filter_by(id=session_data.id).first()
            if not federated_session:
                raise ValueError(f"FederatedSession with ID {session_data.id} not found.")
            
            ################# Testing start
            results = test.start_test(json.loads(federated_session.global_parameters))
            print("Global test results: ", results)
            ################## Testing end
            # Reset client_parameters to an empty JSON object
            federated_session.client_parameters = "{}"  # Empty JSON object
            db.commit()  # Save the reset to the database
            print(f"Client parameters reset after Round {i}.")

        # Save test results for future reference
        """ Yashvir: here you can delete the data of this session from the federated_sessions dictionary after saving the results 
            , saved results contains session_data and test_results across all rounds
        """
    test.save_test_results()
    print("##########################Training Ends####################################")


async def wait_for_price_confirmation(federated_manager: FederatedLearning, session_id: str, timeout: int = 300):
    """
    Asynchronously waits for the client to accept the price before proceeding with federated learning.

    Args:
        federated_manager (FederatedLearning): The federated learning manager handling sessions.
        session_id (str): The ID of the federated learning session.
        timeout (int): Maximum time (in seconds) to wait for the price confirmation. Default is 5 minutes.

    Returns:
        bool: True if the client accepted the price, False if timeout occurred.
    """
    start_time = asyncio.get_event_loop().time()  # Get async time to prevent blocking

    while True:
        session_data = federated_manager.get_session(session_id)

        if session_data.training_status == 2:  # Status 2 means price was accepted
            print(f"✅ Client accepted the price for session {session_id}. Proceeding with training.")
            return True

        # Check for timeout
        if asyncio.get_event_loop().time() - start_time > timeout:
            print(f"⏳ Timeout: Client did not confirm the price within {timeout} seconds. Aborting session {session_id}.")
            return False

        print("⌛ Waiting for client price confirmations... (Training Status: 1)")
        await asyncio.sleep(5)  # Non-blocking sleep

async def wait_for_client_confirmation(federated_manager: FederatedLearning, session_id: int):
    all_ready_for_training = False

    while not all_ready_for_training:
        session_data = federated_manager.get_session(session_id)
        await asyncio.sleep(5)
        now = datetime.now()
        all_ready_for_training = all(client.status != 1 for client in session_data.clients) and session_data.wait_till < now
        
        print("Waiting for client confirmations....Stage 1")

    print("All Clients have taken their decision.")

class MessageType:
    GET_MODEL_PARAMETERS_START_BACKGROUND_PROCESS = "get_model_parameters_start_background_process"
    START_TRAINING = "start_training"

async def send_model_configs_and_wait_for_confirmation(federated_manager: FederatedLearning, session_id: int):

    session_data = federated_manager.get_session(session_id)
    interested_clients = [client.user_id for client in session_data.clients if client.status == 2]
    
    model_config = session_data.federated_info
    
    message = {
        "type": MessageType.GET_MODEL_PARAMETERS_START_BACKGROUND_PROCESS,
        "data": model_config,
        "session_id": session_data.id
    }

    with Session(engine) as db:
        add_notifications_for(db, message, interested_clients)

    # Wait for all clients to confirm they have started their background process
    await wait_for_all_clients_to_stage_four(session_data)


# async def send_message_with_type(client: FederatedSessionClient, message_type: str, data: dict, session_data: FederatedSession):
#     message = {
#         "type": message_type,
#         "data": data,
#         "session_id": session_data.id
#     }

#     json_message = json.dumps(message)

#     print("json model sent before the training signal: ", json_message)
#     print(f"client id {client.id}")
    

async def wait_for_all_clients_to_stage_four(session_data: FederatedSession):
    # Implement the logic to wait for all clients to confirm that they have started background process
    # session_data = federated_manager.federated_sessions[session_id]
    # interested_clients = [client for client in session_data.clients if client.status == 2]
    with Session(engine) as db:
        while True:
            # Expire all objects in the session to force reloading
            db.expire_all()
            
            interested_clients = db.query(FederatedSessionClient).filter_by(session_id = session_data.id).all()
            all_clients_ready = True
            for client in interested_clients:
                if client.local_model_id == None:
                    all_clients_ready = False
                    break

            if all_clients_ready:
                print("All sent local model id")
                break
            else:
                await asyncio.sleep(5)
                print("Waiting for all clients to send local model id.")


async def send_training_signal_and_wait_for_clients_training(federated_manager: FederatedLearning, session_id: str):
    session_data = federated_manager.get_session(session_id)
    interested_clients = [client.user_id for client in session_data.clients if client.status == 4]
    
    model_config = session_data.federated_info
    with Session(engine) as db:
        for client in session_data.clients:
            if client.user_id in interested_clients:
                # Get local_model_id directly from session_data.clients
                local_model_id = client.local_model_id

                # Create a customized message for the client
                message = {
                    "type": MessageType.START_TRAINING,
                    "data": {
                        "model_config": model_config,
                        "local_model_id": local_model_id  # Wrap both in the `data` key
                    },
                    "session_id": session_data.id,
                }
                add_notifications_for_user(db, client.user_id, message)
        
    # Run the wait_for_all_clients_to_local_training task in the background
    await wait_for_all_clients_to_local_training(session_data)
    

async def wait_for_all_clients_to_local_training(session_data: FederatedSession):
    # Implement the logic to wait for all clients to confirm that they have started background process
    # session_data = federated_manager.federated_sessions[session_id]
    # interested_clients = [client for client in session_data.clients if client.status == 2]
    with Session(engine) as db:
        session = db.query(FederatedSession).filter(FederatedSession.id == session_data.id).first()
        
        if not session:
            raise ValueError(f"FederatedSession with ID {session.id} not found.")
        
        num_interested_clients = len(session.clients)
        print("Error Check: Total Interested Clients:", num_interested_clients)
        
        while True:
            # Expire all objects in the session to force reloading
            db.expire_all()
            session = db.query(FederatedSession).filter(FederatedSession.id == session_data.id).first()
            if not session:
                raise ValueError(f"FederatedSession with ID {session_data.id} not found (possibly deleted).")

            
            # Deserialize client_parameters from JSON to Python dict
            client_parameters = json.loads(session.client_parameters) if session.client_parameters else {}


            # Count clients with local model parameters submitted
            num_clients_with_local_models = len([
                client_id for client_id, params in client_parameters.items()
            ])
            
            
            print("Client Parameters:", num_clients_with_local_models)  # Debugging
            
            print(f"Progress: {num_clients_with_local_models}/{num_interested_clients} clients ready.")
            # Check if all interested clients have submitted their parameters
            if num_clients_with_local_models >= num_interested_clients:
                print("All Local Models Trained and Received.")
                break
            else:
                await asyncio.sleep(5)
                print(f"Waiting for {num_interested_clients - num_clients_with_local_models} more clients.")

