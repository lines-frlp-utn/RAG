import os

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, ForeignKey, Integer, String, create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

app = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:password@mysql_db:3306/usuarios_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserCreateDTO(BaseModel):
    identifier: str
    password: str  # puede ser vacio si es OAuth
    role_name: str
    auth_provider: str = "credentials"
    email: str = None
    picture: str


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
    password = Column(String(50), nullable=True)  # Puede ser NULL si es OAuth
    auth_provider = Column(String(50), nullable=False, default="credentials")
    email = Column(String(50), unique=True, nullable=True)
    picture = Column(String(100), nullable=True)
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


def add_column_if_not_exists(db: Session, table_name: str, column_name: str, column_type: str):
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    if column_name not in columns:
        db.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"))
        db.commit()


@app.on_event("startup")
def startup_event():
    db = SessionLocal()
    init_roles(db)
    add_column_if_not_exists(db, "usuarios", "email", "VARCHAR(50)")
    add_column_if_not_exists(db, "usuarios", "auth_provider", "VARCHAR(50)")
    add_column_if_not_exists(db, "usuarios", "picture", "VARCHAR(100)")
    db.close()


@app.get("/users/login/{identifier}")
async def user_exists(identifier: str, password=str, db: Session = Depends(get_db)):
    user = (
        db.query(Usuario)
        .filter(Usuario.identifier == identifier)
        .filter(Usuario.password == password)
        .first()
    )
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
    new_user = Usuario(
        identifier=user.identifier,
        role_id=role.id,
        password=user.password or "",
        auth_provider=user.auth_provider,
        email=user.email,
        picture=user.picture,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return UserCreateDTO(
        identifier=new_user.identifier,
        role_name=role.name,
        password=new_user.password,
        auth_provider=user.auth_provider,
        email=new_user.email,
        picture=new_user.picture,
    )


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
    return UserCreateDTO(
        identifier=user.identifier, role_name=new_role.name, password=user.password
    )
