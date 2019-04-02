ALTER TABLE documents_correspondent ALTER COLUMN name TYPE text;
ALTER TABLE documents_correspondent ALTER COLUMN slug TYPE text;
ALTER TABLE documents_correspondent ALTER COLUMN match TYPE text;
ALTER TABLE documents_tag ALTER COLUMN name TYPE text;
ALTER TABLE documents_tag ALTER COLUMN slug TYPE text;
ALTER TABLE documents_tag ALTER COLUMN match TYPE text;
ALTER TABLE documents_document ALTER COLUMN title TYPE text;
ALTER TABLE documents_document ALTER COLUMN title TYPE text;

DROP INDEX documents_document_content_aa150741;
DROP INDEX documents_document_content_aa150741_like;

ALTER TABLE django_admin_log ALTER COLUMN object_repr TYPE text;

CREATE EXTENSION pg_trgm;
DROP INDEX documents_document_content;
CREATE INDEX documents_document_content ON documents_document USING GIN (upper(content) gin_trgm_ops);
CREATE INDEX documents_correspondent_name ON documents_correspondent USING GIN (upper(name) gin_trgm_ops);
CREATE INDEX documents_document_title ON documents_document USING GIN (upper(title) gin_trgm_ops);
CREATE INDEX documents_tag_name ON documents_tag USING GIN (upper(name) gin_trgm_ops);
