from sqlalchemy.orm import Session
from src.api.db import User, engine
from src.api.services.auth import AuthService

def create_default_user():
    username = "mlops"
    password = "mlops"
    full_name = "Admin"

    with Session(engine) as session:
        existing_user = session.query(User).filter(User.username == username).first()
        if existing_user:
            print("✅ Utilisateur admin déjà existant.")
            return

        # Hash correctement le mot de passe
        hashed_password = AuthService.hash_password(password)

        new_user = User(username=username, hashed_password=hashed_password, full_name=full_name)
        session.add(new_user)
        session.commit()
        print("✅ Utilisateur admin créé avec succès.")

if __name__ == "__main__":
    create_default_user()
