"""
DataMind AI — Data Analyst Agent
==================================
Converts natural language questions into SQL queries,
executes them, and returns human-readable answers.

Flow: NL Question → SQL → Execute → Summarize → Answer
"""

import os
import sqlite3
import logging
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


NL2SQL_PROMPT = """You are an expert SQL analyst. Convert the user's question to a SQL query.

DATABASE SCHEMA:
{schema}

RULES:
1. Return ONLY the SQL query, nothing else.
2. Use proper SQL syntax for SQLite.
3. Always add LIMIT 100 unless the user asks for all records.
4. Use COUNT, SUM, AVG, GROUP BY where appropriate for aggregations.
5. If the question cannot be answered with SQL, respond: CANNOT_ANSWER

USER QUESTION: {question}

SQL QUERY:"""

SUMMARIZE_PROMPT = """You are a data analyst. Summarize the SQL query results in plain English.

ORIGINAL QUESTION: {question}
SQL QUERY EXECUTED: {sql}
QUERY RESULTS (first 20 rows):
{results}

Provide a clear, concise answer to the original question based on these results.
Include key numbers and insights. Be specific.

ANSWER:"""


class DataAnalystAgent:
    """
    AI-powered data analyst that translates natural language
    questions to SQL and executes them against a database.
    """

    def __init__(
        self,
        db_path: str = "data/sample.db",
        llm_provider: str = "openai",
        model: str = "gpt-4o",
        max_retries: int = 3,
    ):
        self.db_path      = db_path
        self.llm_provider = llm_provider
        self.model        = model
        self.max_retries  = max_retries
        self._llm         = None
        self._schema      = None

    def _get_llm(self):
        if self._llm is None:
            from openai import OpenAI
            self._llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._llm

    def _call_llm(self, prompt: str) -> str:
        llm = self._get_llm()
        response = llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def _get_schema(self) -> str:
        """Extract schema from the database."""
        if self._schema:
            return self._schema

        schema_parts = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                df = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", conn)
                cols = ", ".join(
                    f"{col} ({str(df[col].dtype)})"
                    for col in df.columns
                )
                schema_parts.append(f"Table: {table}\nColumns: {cols}")
                schema_parts.append(f"Sample row: {df.iloc[0].to_dict() if len(df) > 0 else 'empty'}")

        self._schema = "\n\n".join(schema_parts)
        return self._schema

    def _generate_sql(self, question: str) -> str:
        """Generate SQL from natural language question."""
        schema = self._get_schema()
        prompt = NL2SQL_PROMPT.format(schema=schema, question=question)
        sql = self._call_llm(prompt)

        # Clean up the SQL
        sql = sql.strip().strip("```sql").strip("```").strip()
        return sql

    def _execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return results as DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(sql, conn)

    def _fix_sql(self, sql: str, error: str, question: str) -> str:
        """Ask LLM to fix broken SQL."""
        fix_prompt = f"""The following SQL query has an error. Fix it.

ORIGINAL QUESTION: {question}
BROKEN SQL: {sql}
ERROR: {error}

Return ONLY the corrected SQL query.

FIXED SQL:"""
        fixed = self._call_llm(fix_prompt)
        return fixed.strip().strip("```sql").strip("```").strip()

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the full NL → SQL → Answer pipeline.

        Args:
            question: Natural language question about the data

        Returns:
            Dict with answer, sql, results_df, error
        """
        logger.info(f"Data Analyst Agent: {question[:80]}")

        sql = self._generate_sql(question)

        if sql == "CANNOT_ANSWER":
            return {
                "answer": "This question cannot be answered with the available data.",
                "sql": None,
                "results": None,
                "error": None,
            }

        # Try executing with retries
        results_df = None
        error = None
        for attempt in range(self.max_retries):
            try:
                results_df = self._execute_sql(sql)
                error = None
                break
            except Exception as e:
                error = str(e)
                logger.warning(f"SQL attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    sql = self._fix_sql(sql, error, question)

        if error:
            return {
                "answer": f"I encountered an error executing the query: {error}",
                "sql": sql,
                "results": None,
                "error": error,
            }

        # Summarize results
        results_preview = results_df.head(20).to_string(index=False)
        summary_prompt  = SUMMARIZE_PROMPT.format(
            question=question,
            sql=sql,
            results=results_preview,
        )
        answer = self._call_llm(summary_prompt)

        return {
            "answer":  answer,
            "sql":     sql,
            "results": results_df,
            "error":   None,
            "row_count": len(results_df),
        }
