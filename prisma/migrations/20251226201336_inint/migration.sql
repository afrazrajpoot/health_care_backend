-- AlterTable
ALTER TABLE "tasks" ADD COLUMN     "type" TEXT DEFAULT 'internal';

-- CreateIndex
CREATE INDEX "tasks_type_idx" ON "tasks"("type");
