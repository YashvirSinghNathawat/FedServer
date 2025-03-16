from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, Float, String, event
from sqlalchemy.orm import declared_attr, relationship, Session, with_loader_criteria
from .Base import Base
import os

load_dotenv()


class TimestampMixin:
    @declared_attr
    def createdAt(cls):
        return Column(DateTime, default=lambda: datetime.now())
    @declared_attr
    def updatedAt(cls):
        return Column(DateTime, default=lambda: datetime.now(), onupdate=lambda: datetime.now())

    @declared_attr
    def deletedAt(cls):
        return Column(DateTime, nullable=True)

@event.listens_for(Session, "do_orm_execute")
def filter_soft_deleted(execute_state):
    if not execute_state.is_column_load and not execute_state.is_relationship_load:
        if not execute_state.execution_options.get("include_deleted", False):
            execute_state.statement = execute_state.statement.options(
                with_loader_criteria(
                    TimestampMixin,
                    lambda cls: cls.deletedAt.is_(None),
                    include_aliases=True
                )
            )

class FederatedSession(TimestampMixin, Base):
    __tablename__ = 'federated_sessions'
    
    id = Column(Integer, primary_key=True, index=True)
    federated_info = Column(JSON, nullable=False)
    admin_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    curr_round = Column(Integer, default=1, nullable=False)
    max_round = Column(Integer, default=3, nullable=False)
    global_parameters = Column(JSON, default='[]', nullable=False)
    session_price = Column(Float, default= 0, nullable=True)
    # 1 for server waiting for admin to price, 2 for server waiting for all clients and 3 for training starts, 4 for completed
    training_status = Column(Integer, default=1, nullable=False) 
    client_parameters = Column(JSON, default='{}', nullable=False)
    # Wait Time
    wait_till = Column(DateTime, default=lambda: datetime.now() + timedelta(minutes=int(os.getenv('SESSION_WAIT_MINUTES'))))
    
    admin = relationship("User", back_populates="federated_sessions")
    clients = relationship('FederatedSessionClient', back_populates='session')
    
    def as_dict(self):
        return {
            "id": self.id,
            "federated_info": self.federated_info,
            "admin_id": self.admin_id,
            "curr_round": self.curr_round,
            "max_round": self.max_round,
            "global_parameters": self.global_parameters,
            "training_status": self.training_status,
            "client_parameters": self.client_parameters,
            "wait_till": self.wait_till.isoformat() if self.wait_till else None,  # Convert DateTime to ISO format
            "admin": self.admin.as_dict() if self.admin else None,               # Call as_dict on the related User
            "clients": [client.as_dict() for client in self.clients]             # Call as_dict on related clients
        }
    
class FederatedSessionClient(TimestampMixin, Base):
    __tablename__ = 'federated_session_clients'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    session_id = Column(Integer, ForeignKey('federated_sessions.id'), nullable=False)
    status = Column(Integer, default=1, nullable=False) # Status values: 1 (not responded), 2 (accepted), 3 (rejected)
    ip = Column(String, nullable=False)
    local_model_id = Column(String, nullable=True)
    
    user = relationship('User', back_populates="federated_session_clients")
    session = relationship('FederatedSession', back_populates='clients')
