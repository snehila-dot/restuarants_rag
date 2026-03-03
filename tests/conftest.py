"""Pytest configuration and fixtures."""

import asyncio
import uuid
from collections.abc import AsyncGenerator, Generator
from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import get_session
from app.main import app
from app.models.restaurant import Base, Restaurant

# Test database URL (in-memory SQLite)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Create test engine
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)

# Create test session factory
TestSessionLocal = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def setup_database() -> AsyncGenerator[None, None]:
    """Create test database tables before each test."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    """Provide database session for tests."""
    async with TestSessionLocal() as session:
        yield session


@pytest.fixture
async def client(session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Provide test client with overridden database session."""

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield session

    app.dependency_overrides[get_session] = override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
async def sample_restaurants(session: AsyncSession) -> list[Restaurant]:
    """Create sample restaurants in test database."""
    restaurants = [
        Restaurant(
            id=uuid.uuid4(),
            name="Test Italian Restaurant",
            address="Test Street 1, Graz",
            cuisine=["Italian", "Pizza"],
            price_range="€€",
            rating=4.5,
            review_count=100,
            features=["outdoor_seating"],
            summary="A test Italian restaurant",
            data_sources=["test"],
            last_verified=datetime.utcnow(),
        ),
        Restaurant(
            id=uuid.uuid4(),
            name="Test Vegan Cafe",
            address="Test Street 2, Graz",
            cuisine=["Vegan", "Vegetarian"],
            price_range="€",
            rating=4.7,
            review_count=50,
            features=["vegan_options", "wifi"],
            summary="A test vegan cafe",
            data_sources=["test"],
            last_verified=datetime.utcnow(),
        ),
        Restaurant(
            id=uuid.uuid4(),
            name="Test Austrian Traditional",
            address="Herrengasse 10, Graz",
            cuisine=["Austrian"],
            price_range="€€€",
            rating=4.3,
            review_count=200,
            features=["reservations", "serves_wine", "outdoor_seating"],
            summary="Traditional Austrian fine dining",
            opening_hours={
                "monday": "11:00-22:00",
                "tuesday": "11:00-22:00",
                "wednesday": "11:00-22:00",
                "thursday": "11:00-22:00",
                "friday": "11:00-23:00",
                "saturday": "17:00-23:00",
                "sunday": "closed",
            },
            latitude=47.0717,
            longitude=15.4377,
            data_sources=["test"],
            last_verified=datetime.utcnow(),
        ),
        Restaurant(
            id=uuid.uuid4(),
            name="Test Family Pizza",
            address="Lendplatz 5, Graz",
            cuisine=["Italian", "Pizza"],
            price_range="€",
            rating=4.0,
            review_count=80,
            features=[
                "good_for_children",
                "children_menu",
                "good_for_groups",
                "delivery",
            ],
            summary="Family-friendly pizzeria",
            opening_hours={
                "monday": "10:00-22:00",
                "tuesday": "10:00-22:00",
                "wednesday": "10:00-22:00",
                "thursday": "10:00-22:00",
                "friday": "10:00-23:00",
                "saturday": "10:00-23:00",
                "sunday": "12:00-21:00",
            },
            latitude=47.0740,
            longitude=15.4310,
            data_sources=["test"],
            last_verified=datetime.utcnow(),
        ),
    ]

    for restaurant in restaurants:
        session.add(restaurant)

    await session.commit()
    return restaurants
