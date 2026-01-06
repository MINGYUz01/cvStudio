"""
pytest配置文件
定义共享fixtures和测试配置
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def db_session():
    """数据库会话fixture"""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def test_user(db_session):
    """测试用户fixture"""
    from app.models.user import User
    from app.core.security import hash_password

    # 创建测试用户
    user = User(
        username="test_user",
        email="test@example.com",
        password_hash=hash_password("test123")
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    yield user

    # 清理
    db_session.delete(user)
    db_session.commit()


@pytest.fixture
def test_dataset(db_session, test_user):
    """测试数据集fixture"""
    from app.models.dataset import Dataset

    dataset = Dataset(
        name="test_dataset",
        path="data/datasets/test",
        format="yolo",
        num_images=100,
        num_classes=10,
        classes=["class1", "class2"],
        is_active="active",
        user_id=test_user.id
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)

    yield dataset

    # 清理
    db_session.delete(dataset)
    db_session.commit()


@pytest.fixture
def test_model(db_session, test_user):
    """测试模型fixture"""
    from app.models.model import Model

    model = Model(
        name="test_model",
        graph_json={},
        code_path="models/test.py",
        template_tag="base",
        user_id=test_user.id
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    yield model

    # 清理
    db_session.delete(model)
    db_session.commit()


@pytest.fixture
def authenticated_client(test_user):
    """已认证的测试客户端fixture"""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.security import create_access_token

    client = TestClient(app)

    # 设置认证token
    token = create_access_token(data={"sub": test_user.username})
    client.headers["Authorization"] = f"Bearer {token}"

    return client
