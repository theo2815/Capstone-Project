ALTER TABLE photo_face_persons
    ADD COLUMN provider varchar(16) NOT NULL DEFAULT 'UNKNOWN'
        CHECK (provider IN ('UNKNOWN', 'AI_API', 'REKOGNITION'));

-- Writer completion and confirmed biometric erasure are separate facts.
ALTER TABLE photo_work_outputs ADD COLUMN faces_erased_at timestamptz;
