"""LLM service for generating natural language responses."""

import logging

from openai import AsyncOpenAI

from app.config import settings
from app.models.restaurant import Restaurant

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = AsyncOpenAI(api_key=settings.openai_api_key)


SYSTEM_PROMPT_EN = """You are a helpful assistant for finding restaurants in Graz, Austria.

CRITICAL RULES:
1. You ONLY provide information from the restaurant data given to you
2. NEVER make up or hallucinate restaurant information
3. NEVER suggest restaurants outside of Graz
4. If the data doesn't have specific information (like opening hours), say so honestly
5. Keep responses concise and factual
6. Temperature is set very low - be deterministic and accurate

Your task is to present the provided restaurant information in a natural, conversational way while answering the user's question."""

SYSTEM_PROMPT_DE = """Du bist ein hilfreicher Assistent zum Finden von Restaurants in Graz, Österreich.

WICHTIGE REGELN:
1. Du gibst NUR Informationen aus den dir bereitgestellten Restaurantdaten
2. NIEMALS erfundene oder halluzinierte Restaurantinformationen
3. NIEMALS Restaurants außerhalb von Graz vorschlagen
4. Wenn die Daten bestimmte Informationen nicht enthalten (z.B. Öffnungszeiten), ehrlich sagen
5. Antworten kurz und sachlich halten
6. Temperatur ist sehr niedrig - sei deterministisch und genau

Deine Aufgabe ist es, die bereitgestellten Restaurantinformationen auf natürliche, gesprächige Weise zu präsentieren und dabei die Frage des Benutzers zu beantworten."""


def format_restaurant_data(restaurants: list[Restaurant]) -> str:
    """
    Format restaurant data for inclusion in LLM prompt.

    Args:
        restaurants: List of Restaurant objects

    Returns:
        Formatted string with restaurant information
    """
    if not restaurants:
        return "No restaurants found matching the criteria."

    formatted = []
    for i, restaurant in enumerate(restaurants, 1):
        info = [
            f"{i}. {restaurant.name}",
            f"   Address: {restaurant.address}",
            f"   Cuisine: {', '.join(restaurant.cuisine)}",
            f"   Price Range: {restaurant.price_range}",
        ]

        if restaurant.rating:
            info.append(
                f"   Rating: {restaurant.rating}/5.0 ({restaurant.review_count} reviews)"
            )

        if restaurant.phone:
            info.append(f"   Phone: {restaurant.phone}")

        if restaurant.website:
            info.append(f"   Website: {restaurant.website}")

        if restaurant.features:
            info.append(f"   Features: {', '.join(restaurant.features)}")

        if restaurant.opening_hours:
            hours_str = ", ".join(
                [f"{day}: {hours}" for day, hours in restaurant.opening_hours.items()]
            )
            info.append(f"   Hours: {hours_str}")

        if restaurant.summary:
            info.append(f"   Summary: {restaurant.summary}")

        if restaurant.menu_items:
            menu_lines = []
            for item in restaurant.menu_items[:10]:  # Cap at 10 items
                entry = f"      - {item.name}"
                if item.price_text:
                    entry += f" ({item.price_text})"
                elif item.price is not None:
                    entry += f" (€{item.price:.2f})"
                if item.category and item.category != "Other":
                    entry += f" [{item.category}]"
                menu_lines.append(entry)
            info.append("   Menu highlights:")
            info.extend(menu_lines)

        if restaurant.menu_url:
            info.append(f"   Full menu: {restaurant.menu_url}")

        formatted.append("\n".join(info))

    return "\n\n".join(formatted)


async def generate_response(
    user_message: str,
    restaurants: list[Restaurant],
    language: str = "en",
    location_hint: str = "",
) -> str:
    """
    Generate a natural language response using the LLM.

    Args:
        user_message: The user's original question
        restaurants: List of relevant restaurants
        language: Response language ('en' or 'de')
        location_hint: Optional location context (e.g. " near jakominiplatz")

    Returns:
        Generated response text
    """
    system_prompt = SYSTEM_PROMPT_DE if language == "de" else SYSTEM_PROMPT_EN
    restaurant_data = format_restaurant_data(restaurants)

    location_ctx = ""
    if location_hint:
        location_ctx = (
            f"\nNote: These restaurants are located{location_hint}. "
            "Mention their proximity to this area in your response."
        )

    user_prompt = f"""User question: {user_message}

Available restaurant data:
{restaurant_data}
{location_ctx}
Please answer the user's question using ONLY the information provided above. Be concise and helpful."""

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.llm_temperature,
            max_tokens=500,
        )

        return response.choices[0].message.content or "I couldn't generate a response."

    except Exception as e:
        logger.error(f"LLM generation error: {e}")

        # Fallback response
        if language == "de":
            return f"Ich habe {len(restaurants)} Restaurant(s) für Sie gefunden. Bitte sehen Sie sich die Details unten an."
        else:
            return f"I found {len(restaurants)} restaurant(s) for you. Please see the details below."
