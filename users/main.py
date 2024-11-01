from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import Column, Integer, String, create_engine, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import os
from pydantic import BaseModel

app = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:password@mysql_db:3306/usuarios_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserCreateDTO(BaseModel):
    identifier: str
    role_name: str

class RoleResponseDTO(BaseModel):
    exists: bool
    role_name: str = None

class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)

    users = relationship("Usuario", back_populates="role")

class Usuario(Base):
    __tablename__ = "usuarios"
    id = Column(Integer, primary_key=True, index=True)
    identifier = Column(String(50), unique=True, index=True, nullable=False)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)

    role = relationship("Role", back_populates="users")

# Crear las tablas que no estan creadas
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_roles(db: Session):
    roles = ["ADMIN", "CLIENTE"]
    for role_name in roles:
        role = db.query(Role).filter(Role.name == role_name).first()
        if not role:
            db.add(Role(name=role_name))
    db.commit()

@app.on_event("startup")
def startup_event():
    db = SessionLocal()
    init_roles(db)
    db.close()

@app.get("/users/exists/{identifier}")
async def user_exists(identifier: str, db: Session = Depends(get_db)):
    user = db.query(Usuario).filter(Usuario.identifier == identifier).first()
    if user:
        userResponse = RoleResponseDTO(exists=True, role_name=user.role.name)
        return userResponse
    return RoleResponseDTO(exists=False)


@app.post("/users/create/")
async def create_user(user: UserCreateDTO, db: Session = Depends(get_db)):
    role = db.query(Role).filter(Role.name == user.role_name).first()
    if not role:
        raise HTTPException(status_code=400, detail="Role not found")
    existing_user = db.query(Usuario).filter(Usuario.identifier == user.identifier).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    new_user = Usuario(identifier=user.identifier, role_id=role.id)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return UserCreateDTO(identifier=new_user.identifier, role_name=role.name)

@app.patch("/users/role/{identifier}")
async def update_user_role(identifier: str, role: str, db: Session = Depends(get_db)):
    user = db.query(Usuario).filter(Usuario.identifier == identifier).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    new_role = db.query(Role).filter(Role.name == role).first()
    if not new_role:
        raise HTTPException(status_code=400, detail="Role not found")
    user.role_id = new_role.id
    db.commit()
    db.refresh(user)
    return UserCreateDTO(identifier=user.identifier, role_name=new_role.name)