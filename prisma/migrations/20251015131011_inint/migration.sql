-- AlterTable
ALTER TABLE "FailDocs" ADD COLUMN     "blobPath" TEXT,
ADD COLUMN     "fileHash" TEXT,
ADD COLUMN     "fileName" TEXT,
ADD COLUMN     "gcsFileLink" TEXT;
