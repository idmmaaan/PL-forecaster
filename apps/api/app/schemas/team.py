from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class TeamSummary(BaseModel):
    # The published field stays `name` while the column is `canonical_name`, so
    # the alias keeps the API contract stable across the rename.
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    name: str = Field(validation_alias=AliasChoices("canonical_name", "name"))
    crest_url: str | None = None
